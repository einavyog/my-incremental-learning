''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from __future__ import print_function

import copy
import logging

import numpy as np
import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

import model
from  torch.autograd import backward
from torch.autograd.gradcheck import zero_gradients
import os
from utils import matmul, tensormul
import time

logger = logging.getLogger('iCARL')


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator=None,
                 model_jm=None, optimizer_jm=None):
        self.train_data_iterator = trainDataIterator
        self.test_data_iterator = testDataIterator
        self.model = model
        self.args = args
        self.dataset = dataset
        self.train_loader = self.train_data_iterator.dataset
        self.older_classes = []
        self.optimizer = optimizer
        self.model_fixed = copy.deepcopy(self.model)
        self.active_classes = []
        for param in self.model_fixed.parameters():
            param.requires_grad = False

        self.model_jm = model_jm
        self.optimizer_jm = optimizer_jm
        self.model_fixed_jm = copy.deepcopy(self.model_jm)
        self.models_jm = []
        self.decay_jm = args.jm_decay
        if self.model_fixed_jm is not None:
            for param in self.model_fixed_jm.parameters():
                param.requires_grad = False

        self.models = []
        self.current_lr = args.lr
        self.current_lr_jm = args.lr
        self.all_classes = list(range(dataset.classes))
        self.all_classes.sort(reverse=True)
        self.left_over = []
        self.ideal_iterator = ideal_iterator
        self.model_single = copy.deepcopy(self.model)
        self.optimizer_single = None

        self.seed = args.seed
        self.projection_dim = args.projection_dim

        logger.warning("Shuffling turned off for debugging")
        # random.seed(args.seed)
        # random.shuffle(self.all_classes)


class Trainer(GenericTrainer):

    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator=None,
                 model_jm=None, optimizer_jm=None):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator, model_jm, optimizer_jm)
        self.dynamic_threshold = np.ones(self.dataset.classes, dtype=np.float64)
        self.gradient_threshold_unreported_experiment = np.ones(self.dataset.classes, dtype=np.float64)

    def update_lr(self, epoch, use_jm=False):

        for temp in range(0, len(self.args.schedule)):
            if self.args.schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    logger.debug("Changing learning rate from %0.4f to %0.4f", self.current_lr,
                                 self.current_lr * self.args.gammas[temp])
                    self.current_lr *= self.args.gammas[temp]

        if use_jm:
            for temp in range(0, len(self.args.schedule)):

                if self.args.schedule[temp] == epoch:

                    for param_group in self.optimizer_jm.param_groups:
                        self.current_lr_jm = param_group['lr']
                        param_group['lr'] = self.current_lr_jm * self.args.gammas[temp]
                        logger.debug("Changing learning rate from %0.4f to %0.4f for jm model", self.current_lr_jm,
                                     self.current_lr_jm * self.args.gammas[temp])
                        self.current_lr_jm *= self.args.gammas[temp]

    def increment_classes(self, class_group):
        '''
        Add classes starting from class_group to class_group + step_size 
        :param class_group: 
        :return: N/A. Only has side-affects 
        '''
        for temp in range(class_group, class_group + self.args.step_size):
            pop_val = self.all_classes.pop()
            self.train_data_iterator.dataset.add_class(pop_val)
            self.ideal_iterator.dataset.add_class(pop_val)
            self.test_data_iterator.dataset.add_class(pop_val)
            self.left_over.append(pop_val)

    def limit_class(self, n, k, herding=True):
        if not herding:
            self.train_loader.limit_class(n, k)
        else:
            self.train_loader.limit_class_and_sort(n, k, self.model_fixed)
        if n not in self.older_classes:
            self.older_classes.append(n)

    def reset_dynamic_threshold(self):
        '''
        Reset the threshold vector maintaining the scale factor. 
        Important to set this to zero before every increment. 
        setupTraining() also does this so not necessary to call both. 
        :return: 
        '''
        threshTemp = self.dynamic_threshold / np.max(self.dynamic_threshold)
        threshTemp = ['{0:.4f}'.format(i) for i in threshTemp]

        threshTemp2 = self.gradient_threshold_unreported_experiment / np.max(
            self.gradient_threshold_unreported_experiment)
        threshTemp2 = ['{0:.4f}'.format(i) for i in threshTemp2]

        logger.debug("Scale Factor" + ",".join(threshTemp))
        logger.debug("Scale GFactor" + ",".join(threshTemp2))

        self.dynamic_threshold = np.ones(self.dataset.classes, dtype=np.float64)
        self.gradient_threshold_unreported_experiment = np.ones(self.dataset.classes, dtype=np.float64)

    def setup_training(self, use_jm=False):
        self.reset_dynamic_threshold()

        for param_group in self.optimizer.param_groups:
            logger.debug("Setting LR to %0.4f", self.args.lr)
            param_group['lr'] = self.args.lr
            self.current_lr = self.args.lr

        if use_jm:
            for param_group in self.optimizer_jm.param_groups:
                logger.debug("Setting LR to %0.4f for model jm", self.args.lr)
                param_group['lr'] = self.args.lr
                self.current_lr_jm = self.args.lr

        for val in self.left_over:
            self.limit_class(val, int(self.args.memory_budget / len(self.left_over)), not self.args.no_herding)

    def update_frozen_model(self, use_jm=False):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.models.append(self.model_fixed)

        if self.args.random_init:
            logger.warning("Random Initilization of weights at each increment")
            myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)

            if self.args.cuda:
                myModel.cuda()

            self.model = myModel
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                             weight_decay=self.args.decay, nesterov=True)
            self.model.eval()

        if use_jm:
            self.model_jm.eval()
            self.model_fixed_jm = copy.deepcopy(self.model_jm)
            self.model_fixed_jm.eval()
            for param in self.model_fixed_jm.parameters():
                param.requires_grad = False
            self.models_jm.append(self.model_fixed_jm)

            if self.args.random_init:
                logger.warning("JM model random initializes as the model")

                self.model_jm.load_state_dict(copy.deepcopy(self.model.state_dict()))
                self.optimizer_jm = torch.optim.SGD(self.model_jm.parameters(), self.args.lr, momentum=self.args.momentum,
                                                 weight_decay=self.args.decay, nesterov=True)
                self.model_jm.eval()

    def randomInitModel(self, use_jm=False):
        logger.info("Randomly initilizaing model")
        myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
        if self.args.cuda:
            myModel.cuda()
        self.model = myModel
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                         weight_decay=self.args.decay, nesterov=True)
        self.model.eval()

        if use_jm:
            logger.info("Randomly initilizaing model_jm")
            myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
            if self.args.cuda:
                myModel.cuda()
            self.model_jm = myModel
            self.optimizer = torch.optim.SGD(self.model_jm.parameters(), self.args.lr, momentum=self.args.momentum,
                                             weight_decay=self.args.decay, nesterov=True)
            self.model_jm.eval()

    def get_model(self):
        myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)
        if self.args.cuda:
            myModel.cuda()
        optimizer = torch.optim.SGD(myModel.parameters(), self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.decay, nesterov=True)
        myModel.eval()

        self.current_lr = self.args.lr

        self.model_single = myModel
        self.optimizer_single = optimizer
    # 
    # def compute_normalized_jacobian(self, data, use_fixed_model=True, use_model_jm=False,
    #                                 is_norm=True, is_calc_from_embedded=False, random_normal_mat=None):
    # 
    #     jacobian = self.compute_jacobian(data, use_fixed_model, use_model_jm, random_normal_mat)
    # 
    #     if is_norm:
    #         jn = torch.norm(jacobian, dim=(3, 4)).detach()
    #         jn = jn.unsqueeze(3)
    #         jn = jn.unsqueeze(4)
    #         jacobian_norm = jacobian.div(jn.expand_as(jacobian))
    # 
    #         return jacobian_norm
    # 
    #     else:
    #         return jacobian
    # 
    # ################ Function for Jacobian calculation ################
    # def compute_jacobian(self, data, use_fixed_model=True, use_model_jm=False, random_normal_mat=None):
    # 
    #     inputs = Variable(data, requires_grad=True)
    # 
    #     if not use_model_jm:
    #         if use_fixed_model:
    #             output = self.model_fixed(inputs)
    #         else:
    #             output = self.model(inputs)
    #     else:
    #         if use_fixed_model:
    #             output = self.model_fixed_jm.forward(inputs)
    #         else:
    #             output = self.model_jm(inputs)
    # 
    #     output_projection = matmul(random_normal_mat, output)
    # 
    #     grad_output = torch.zeros(*output_projection.size())
    #     if inputs.is_cuda:
    #         grad_output = grad_output.cuda()
    # 
    #     num_classes = output_projection.size()[1]
    # 
    #     jacobian_list = []
    #     # grad_output = torch.zeros(*output.size())
    #     #
    #     # if inputs.is_cuda:
    #     #     grad_output = grad_output.cuda()
    # 
    #     for i in range(num_classes):
    #     #for i in self.older_classes:
    #         zero_gradients(inputs)
    # 
    #         grad_output_curr = grad_output.clone()
    #         grad_output_curr[:, i] = 1
    #         jacobian_list.append(torch.autograd.grad(outputs=output_projection,
    #                                                  inputs=inputs,
    #                                                  grad_outputs=grad_output_curr,
    #                                                  only_inputs=True,
    #                                                  retain_graph=True,
    #                                                  create_graph=not use_fixed_model)[0])
    # 
    #     jacobian = torch.stack(jacobian_list, dim=0)
    # 
    #     return jacobian

    ################ Function for Norm Jacobian calculation ################
    def compute_jacobian(self, data, use_fixed_model=True, use_model_jm=False,
                         is_norm=True, is_calc_from_outputs=False, random_normal_mat=None):

        if is_calc_from_outputs:
            jacobian = self.compute_jacobian_from_outputs(data, use_fixed_model, use_model_jm)
            if self.args.no_projection:
                jacobian_dim = len(self.older_classes)
            else:
                jacobian_dim = self.projection_dim

        else:
            jacobian = self.compute_jacobian_from_embedded(data, use_fixed_model, use_model_jm, random_normal_mat)
            if self.args.no_projection:
                jacobian_dim = 64
            else:
                jacobian_dim = self.projection_dim

        if is_norm:
            a = jacobian
            a = a.permute(1, 0, 2, 3, 4)
            a = a.contiguous().view(self.args.batch_size, -1)
            b = torch.norm(a, dim=1)
            b = b.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
            jacobian_norm = jacobian.div(b.expand_as(jacobian))

            return jacobian_norm

        else:
            return jacobian

    def get_model_outputs(self, inputs, use_model_jm, use_fixed_model):
        if not use_model_jm:
            if use_fixed_model:
                output = self.model_fixed.forward(inputs)
            else:
                output = self.model.forward(inputs)
        else:
            if use_fixed_model:
                output = self.model_fixed_jm.forward(inputs)
            else:
                output = self.model_jm.forward(inputs)

        return output

    ################ Function for Jacobian calculation ################
    def compute_jacobian_from_outputs(self, data, use_fixed_model=True, use_model_jm=False, random_normal_mat=None):

        inputs = Variable(data, requires_grad=True)
        output = self.get_model_outputs(self, inputs, use_model_jm, use_fixed_model)

        if self.args.no_projection:
            output_projection = output
        else:
            output_projection = matmul(random_normal_mat, output)

        grad_output = torch.zeros(*output_projection.size())
        if inputs.is_cuda:
            grad_output = grad_output.cuda()

        jacobian_list = [self.calc_jacobian_loop(inputs, grad_output.clone(), output_projection, use_fixed_model, i)
                         for i in range(output_projection.size()[0])]

        # grad_output = torch.zeros(*output.size())
        # if inputs.is_cuda:
        #     grad_output = grad_output.cuda()
        #
        # jacobian_list = [self.calc_jacobian_loop(inputs, grad_output.clone(), output, use_fixed_model, i)
        #                  for i in range(output.size()[0])]

        # for i in self.older_classes:
        #     zero_gradients(inputs)
        #
        #     grad_output_curr = grad_output.clone()
        #     grad_output_curr[:, i] = 1
        #     jacobian_list.append(torch.autograd.grad(outputs=output,
        #                                              inputs=inputs,
        #                                              grad_outputs=grad_output_curr,
        #                                              only_inputs=True,
        #                                              retain_graph=True,
        #                                              create_graph=not use_fixed_model)[0])

        jacobian = torch.stack(jacobian_list, dim=0)

        return jacobian

    def calc_jacobian_loop(self, inputs, grad_output, output, use_fixed_model, i):
        zero_gradients(inputs)

        grad_output[i, :] = 1
        return torch.autograd.grad(outputs=output,
                                   inputs=inputs,
                                   grad_outputs=grad_output,
                                   only_inputs=True,
                                   retain_graph=True,
                                   create_graph=not use_fixed_model)[0]

    def get_model_outputs_and_embedded_space(self, inputs, use_model_jm, use_fixed_model):

        if not use_model_jm:
            if use_fixed_model:
                output, embedded, x1, x2, x3 = self.model_fixed.forward(inputs, embedding_space=True)
            else:
                output, embedded, x1, x2, x3 = self.model.forward(inputs, embedding_space=True)
        else:
            if use_fixed_model:
                output, embedded, x1, x2, x3 = self.model_fixed_jm.forward(inputs, embedding_space=True)
            else:
                output, embedded, x1, x2, x3 = self.model_jm.forward(inputs, embedding_space=True)

        return output, embedded, x1, x2, x3

    ################ Function for Jacobian calculation ################
    def compute_jacobian_from_embedded(self, data, use_fixed_model=True, use_model_jm=False, random_normal_mat=None):

        inputs = Variable(data, requires_grad=True)
        output, embedded, x1, x2, x3 = self.get_model_outputs_and_embedded_space(inputs, use_model_jm, use_fixed_model)

        if self.args.no_projection:
            embedded_projection = embedded
        else:
            embedded_projection = matmul(random_normal_mat[0], embedded)
            x1_ = matmul(random_normal_mat[1], x1.view(x1.shape[0], x1.shape[1]*x1.shape[2]*x1.shape[3]))
            x2_ = matmul(random_normal_mat[2], x2.view(x2.shape[0], x2.shape[1]*x2.shape[2]*x2.shape[3]))
            x3_ = matmul(random_normal_mat[3], x3.view(x3.shape[0], x3.shape[1]*x3.shape[2]*x3.shape[3]))

        grad_output = torch.zeros(*embedded_projection.size())
        grad_output_x = torch.zeros(*x1_.size())
        if inputs.is_cuda:
            grad_output, grad_output_x = grad_output.cuda(), grad_output_x.cuda()

        jacobian_list_embedded = [self.calc_jacobian_loop(inputs, grad_output.clone(), embedded_projection, use_fixed_model, i)
                         for i in range(embedded_projection.size()[0])]

        jacobian_list_x1 = [self.calc_jacobian_loop(inputs, grad_output_x.clone(), x1_, use_fixed_model, i)
                         for i in range(x1_.size()[0])]

        jacobian_list_x2 = [self.calc_jacobian_loop(inputs, grad_output_x.clone(), x2_, use_fixed_model, i)
                         for i in range(x2_.size()[0])]

        jacobian_list_x3 = [self.calc_jacobian_loop(inputs, grad_output_x.clone(), x3_, use_fixed_model, i)
                         for i in range(x3_.size()[0])]

        # for i in range(embedded_projection.size()[0]):
        #     zero_gradients(inputs)
        #
        #     grad_output_curr = grad_output.clone()
        #     grad_output_curr[i, :] = 1
        #     jacobian_list.append(torch.autograd.grad(outputs=embedded_projection,
        #                                              inputs=inputs,
        #                                              grad_outputs=grad_output_curr,
        #                                              only_inputs=True,
        #                                              retain_graph=True,
        #                                              create_graph=not use_fixed_model)[0])

        jacobian = torch.stack(jacobian_list_embedded, dim=0)

        return jacobian

    def scale_gradient_by_square_of_T(self, myT, use_model_jm):

        # Scale gradient by a factor of square of T.
        # See Distilling Knowledge in Neural Networks by Hinton et.al. for details.
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = param.grad * (myT * myT) * self.args.alpha

        if use_model_jm:
            # TODO: check if square of T is needed here too (Einav)
            for param in self.model_jm.parameters():
                if param.grad is not None:
                    param.grad = param.grad * self.args.alpha

    def cancel_batch_norm_for_increments(self):

            # if self.args.no_bn and len(self.older_classes) > 1:
            if self.args.no_bn:

                for m in self.model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        # m.weight.requires_grad = False
                        # m.bias.requires_grad = False

    def train(self, epoch, use_model_jm=False):

        self.model.train()
        if use_model_jm:
            self.model_jm.train()

        logger.info("Epoch %d", epoch)

        # random_normal_mat = torch.randn(self.args.batch_size, self.projection_dim, self.train_loader.labels.shape[-1])
        embedded_random_mat = torch.randn(self.args.batch_size, self.projection_dim, 64) #TODO: Change to embedded space dim if needed
        x1_random_mat = torch.randn(self.args.batch_size, 2, 16*32*32)
        x2_random_mat = torch.randn(self.args.batch_size, 2, 32*16*16)
        x3_random_mat = torch.randn(self.args.batch_size, 2, 64*8*8)

        if self.args.cuda:
            embedded_random_mat, x1_random_mat, x2_random_mat, x3_random_mat = \
                embedded_random_mat.cuda(), x1_random_mat.cuda(), x2_random_mat.cuda(), x3_random_mat.cuda()

        random_mats = [embedded_random_mat, x1_random_mat, x2_random_mat, x3_random_mat]

        number_of_iterations = 0
        total_loss = 0
        total_classification_loss_jm = 0
        total_loss2 = 0
        total_jacobian_matching_loss = 0
        total_activation_match_loss = 0

        for data, y, target in tqdm(self.train_data_iterator):

            if self.args.batch_size != target.__len__():
                continue

            if 0 == epoch:
                try:
                    bin_count += target.bincount()
                except NameError:
                    bin_count = target.bincount()
                except:
                    print('Couldn''t increase bin_count')

            if self.args.cuda:
                data, target, y = data.cuda(), target.cuda(), y.cuda()

            oldClassesIndices = (target * 0).int()
            for elem in range(0, self.args.unstructured_size):
                oldClassesIndices = oldClassesIndices + (target == elem).int()

            old_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices > 0)).long())
            new_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices == 0)).long())

            self.optimizer.zero_grad()
            if use_model_jm:
                self.optimizer_jm.zero_grad()

            # Use only new classes for normal classification_loss:
            target_normal_loss = target[new_classes_indices]
            data_normal_loss = data[new_classes_indices]

            # Use all of the data for distillation loss:
            target_distillation_loss = y.float()
            data_distillation_loss = data

            # Create y_onehot tensor for normal classification_loss
            y_onehot = torch.FloatTensor(len(target_normal_loss), self.dataset.classes)
            if self.args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target_normal_loss.unsqueeze_(1)
            y_onehot.scatter_(1, target_normal_loss, 1)

            output = self.model(Variable(data_normal_loss))
            self.dynamic_threshold += np.sum(y_onehot.cpu().numpy(), 0)
            loss = F.kl_div(output, Variable(y_onehot))

            if use_model_jm:
                output_jm = self.model_jm(Variable(data_normal_loss))
                classification_loss_jm = F.kl_div(output_jm, Variable(y_onehot))

            myT = self.args.T

            if self.args.no_distill:
                pass

            elif len(self.older_classes) > 0:
                # Get softened labels of the model from a previous version of the model.
                pred2 = self.model_fixed(Variable(data_distillation_loss), T=myT, labels=True).data
                # Softened output of the model
                if myT > 1:
                    output2 = self.model(Variable(data_distillation_loss), T=myT)
                else:
                    output2 = output

                self.dynamic_threshold += (np.sum(pred2.cpu().numpy(), 0)) * (
                        myT * myT) * self.args.alpha
                loss2 = F.kl_div(output2, Variable(pred2))

                loss2.backward(retain_graph=True)

                if use_model_jm:
                    # # Get softened labels of the model from a previous version of the model.
                    # pred2 = self.model_fixed_jm(Variable(data_distillation_loss), T=myT, labels=True).data
                    # # Softened output of the model
                    # if myT > 1:
                    #     output2 = self.model_jm(Variable(data_distillation_loss), T=myT)
                    # else:
                    #     output2 = output_jm
                    #
                    # loss3 = F.kl_div(output2, Variable(pred2))
                    #
                    # loss3.backward(retain_graph=True)

                    # #TODO: Change cpu threshold for jacobian_matching_loss (Einav)
                    jacobian = self.compute_jacobian(data, use_fixed_model=False,
                                                     use_model_jm=use_model_jm,
                                                     is_norm=self.args.norm_jacobian,
                                                     is_calc_from_outputs=self.args.project_outputs,
                                                     random_normal_mat=random_mats)

                    jacobian_model_fixed = self.compute_jacobian(data, use_fixed_model=True,
                                                                 use_model_jm=use_model_jm,
                                                                 is_norm=self.args.norm_jacobian,
                                                                 is_calc_from_outputs=self.args.project_outputs,
                                                                 random_normal_mat=random_mats)

                    jacobian_matching_loss = self.decay_jm*torch.norm(jacobian - jacobian_model_fixed)
                    jacobian_matching_loss.backward(retain_graph=True)

                    _, embedded, x1, x2, x3 = self.get_model_outputs_and_embedded_space(data, use_model_jm, use_fixed_model=True)
                    _, embedded_model_fixed, x1, x2, x3 = self.get_model_outputs_and_embedded_space(data, use_model_jm, use_fixed_model=False)
                    activation_matching_loss = self.args.activation_decay*torch.norm(embedded - embedded_model_fixed)
                    activation_matching_loss.backward(retain_graph=True)


                # Scale gradient by a factor of square of T. See Distilling Knowledge in Neural Networks by Hinton et.al. for details.
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad * (myT * myT) * self.args.alpha

            if len(self.older_classes) == 0 or not self.args.no_nl:
                if not self.args.no_jm_classification or len(self.older_classes) == 0:
                    loss.backward()
                if use_model_jm:
                    if not self.args.no_jm_classification or len(self.older_classes) == 0:
                        classification_loss_jm.backward()

            for param in self.model.named_parameters():
                if "fc.weight" in param[0]:
                    self.gradient_threshold_unreported_experiment *= 0.99
                    self.gradient_threshold_unreported_experiment += np.sum(np.abs(param[1].grad.data.cpu().numpy()), 1)
            self.optimizer.step()
            if use_model_jm:
                self.optimizer_jm.step()

            number_of_iterations += 1
            total_loss += loss.detach()
            total_classification_loss_jm += classification_loss_jm.detach()
            if not self.args.no_distill and len(self.older_classes) > 0:
                total_loss2 += loss2.detach()
                total_jacobian_matching_loss += jacobian_matching_loss.detach()
                total_activation_match_loss += activation_matching_loss.detach()

        total_loss = total_loss/number_of_iterations
        total_classification_loss_jm = total_classification_loss_jm/number_of_iterations
        if not self.args.no_distill and len(self.older_classes) > 0:
            total_loss2 = total_loss2/number_of_iterations
            total_jacobian_matching_loss = total_jacobian_matching_loss/number_of_iterations
            total_activation_match_loss = total_activation_match_loss/number_of_iterations

        if 0 == epoch:
            bin_count_norm = bin_count.float() / bin_count.float().sum()
            logger.debug("bin_count: " + str(bin_count.tolist()))
            logger.debug("bin_count_norm: " + str(bin_count_norm.tolist()))

        if self.args.no_nl:
            self.dynamic_threshold[len(self.older_classes):len(self.dynamic_threshold)] = np.max(self.dynamic_threshold)
            self.gradient_threshold_unreported_experiment[
            len(self.older_classes):len(self.gradient_threshold_unreported_experiment)] = np.max(
                self.gradient_threshold_unreported_experiment)
        else:
            self.dynamic_threshold[0:self.args.unstructured_size] = np.max(self.dynamic_threshold)
            self.gradient_threshold_unreported_experiment[0:self.args.unstructured_size] = np.max(
                self.gradient_threshold_unreported_experiment)

            self.dynamic_threshold[self.args.unstructured_size + len(
                self.older_classes) + self.args.step_size: len(self.dynamic_threshold)] = np.max(
                self.dynamic_threshold)
            self.gradient_threshold_unreported_experiment[self.args.unstructured_size + len(
                self.older_classes) + self.args.step_size: len(self.gradient_threshold_unreported_experiment)] = np.max(
                self.gradient_threshold_unreported_experiment)

        if logger is not None and epoch % 1 == (1 - 1):
            logger.debug("*********CURRENT EPOCH********** : %d", epoch)
            logger.debug("Classification Loss: %0.5f", total_loss)
            if use_model_jm:
                logger.debug("Classification Loss JM: %0.5f", total_classification_loss_jm)

            if not self.args.no_distill and len(self.older_classes) > 0:
                logger.debug("Distillation Loss: %0.5f", total_loss2)
                if use_model_jm:
                    logger.debug("Jacobian Matching Loss: %0.5f", total_jacobian_matching_loss)
                    logger.debug("Activation Matching Loss: %0.5f", total_activation_match_loss)

    def add_model(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        logger.debug("Total Models %d", len(self.models))

    def save_models(self,  file_name, use_model_jm=False):
        torch.save(self.model.state_dict(), file_name + '.pth')
        if use_model_jm:
            torch.save(self.model_jm.state_dict(), file_name + '_jm.pth')

    def load_models(self, pretrained_model=None, pretrained_model_jm=None):

        if pretrained_model:
            pretrain_parameters = torch.load(pretrained_model)
            self.model.load_state_dict(pretrain_parameters)

        if pretrained_model_jm:
            pretrain_parameters = torch.load(pretrained_model_jm)
            self.model_jm.load_state_dict(pretrain_parameters)

    def plot_3d(self, jacobian):

        from mpl_toolkits.mplot3d import Axes3D

        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import numpy as np
        #
        # if 'MNIST' in self.args.dataset:
        #     len_x = jacobian.shape[-2] * jacobian.shape[-1]
        # else:
        #     len_x = jacobian.shape[-2] * jacobian.shape[-3] * jacobian.shape[-1]
        #
        # len_y = jacobian.shape[-5]

        len_x = jacobian.shape[-1]
        len_y = jacobian.shape[-2]

        # for im_num in range(0, jacobian.shape[1]):
        im_num = 0
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(0, len_x)
        Y = np.arange(0, len_y)
        X, Y = np.meshgrid(X, Y)

        Z_tmp = torch.squeeze(jacobian[:, im_num, :, :, :]).cpu()
        Z = np.sum(np.abs(Z_tmp.numpy()), axis=0)
        if 'MNIST' not in self.args.dataset:
            Z = np.sum(Z, axis=0)

        # Z = np.array(Z_tmp.reshape([len_y, len_x]).cpu())

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(-0.01, np.max(Z))
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
        fig.savefig(self.args.dataset + '_' + str(im_num) + '.png', bbox_inches='tight')
        plt.close(fig)

