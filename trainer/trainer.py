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
from utils import matmul

logger = logging.getLogger('iCARL')
np.random.seed(0)

DEBUG = False

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

        self.model_jm = model_jm
        self.optimizer_jm = optimizer_jm
        self.model_fixed_jm = copy.deepcopy(self.model_jm)
        self.models_jm = []
        self.current_lr_jm = args.lr
        self.decay_jm = args.jm_decay

        for param in self.model_fixed.parameters():
            param.requires_grad = False

        self.models = []
        self.current_lr = args.lr
        self.all_classes = list(range(dataset.classes))
        self.all_classes.sort(reverse=True)
        self.left_over = []
        self.ideal_iterator = ideal_iterator
        self.model_single = copy.deepcopy(self.model)
        self.optimizer_single = None

        self.seed = args.seed

        logger.warning("Shuffling turned off for debugging")
        # random.seed(args.seed)
        # random.shuffle(self.all_classes)


class AutoEncoderTrainer(GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)

    def auto_encoder_model(self, noOfFeatures):
        '''
        :param noOfFeatures: No of features of the feature map. This is model dependant so not a constant
        :return: An auto-encoder that reduces the dimensions by a factor of 10. The auto encoder model has the same interface as
        other models implemented in model module.
        '''

        class AutoEncoderModelClass(nn.Module):
            def __init__(self, noOfFeatures):
                super(AutoEncoderModelClass, self).__init__()
                self.featureSize = int(noOfFeatures / 10)
                self.fc1 = nn.Linear(noOfFeatures, int(noOfFeatures / 10))
                self.fc2 = nn.Linear(int(noOfFeatures / 10), noOfFeatures)

            def forward(self, x, feature=False):
                x = F.sigmoid(self.fc1(x))

                if feature:
                    return x
                return self.fc2(x)

        myEncoder = AutoEncoderModelClass(noOfFeatures)

        if self.args.cuda:
            myEncoder.cuda()
        return myEncoder

    def train_auto_encoder(self, xIterator, epochs):
        bar = progressbar.ProgressBar()

        for epoch in range(epochs):

            for batch_idx, (data, target) in bar(enumerate(self.train_data_iterator)):
                pass

    def optimize(self, x, y, optimizer):
        pass


class Trainer(GenericTrainer):

    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator=None,
                 model_jm=None, optimizer_jm=None):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer, ideal_iterator, model_jm, optimizer_jm)
        self.threshold = np.ones(self.dataset.classes, dtype=np.float64)
        self.threshold2 = np.ones(self.dataset.classes, dtype=np.float64)

    def update_lr(self, epoch, use_jm=False):

        for temp in range(0, len(self.args.schedule)):

            if self.args.schedule[temp] == epoch:

                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    logger.debug("Changing learning rate from %0.2f to %0.2f", self.current_lr,
                                 self.current_lr * self.args.gammas[temp])
                    self.current_lr *= self.args.gammas[temp]

        if use_jm:
            for temp in range(0, len(self.args.schedule)):

                if self.args.schedule[temp] == epoch:

                    for param_group in self.optimizer_jm.param_groups:
                        self.current_lr_jm = param_group['lr']
                        param_group['lr'] = self.current_lr_jm * self.args.gammas[temp]
                        logger.debug("Changing learning rate from %0.2f to %0.2f", self.current_lr_jm,
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

    def increment_classes_2(self, start, end):
        '''
        Add classes from start to end (so 10, 30 will add all classes from 10 to 29)
        :param start: Int specifying starting Index. 
        :param end: Int specifying ending Index. 
        :return: 
        '''
        for temp in range(start, end):
            pop_val = self.all_classes.pop()
            self.train_data_iterator.dataset.add_class(pop_val)

            self.ideal_iterator.dataset.add_class(pop_val)
            self.ideal_iterator.dataset.limit_class(pop_val, 0)

            self.test_data_iterator.dataset.add_class(pop_val)
            self.test_data_iterator.dataset.limit_class(pop_val, 0)

    def limit_class(self, n, k, herding=True):

        if not herding:
            self.train_loader.limit_class(n, k)
        else:
            self.train_loader.limit_class_and_sort(n, k, self.model_fixed)

        if n not in self.older_classes:
            self.older_classes.append(n)

    def resetThresh(self):
        '''
        Reset the threshold vector maintaining the scale factor. 
        Important to set this to zero before every increment. 
        setupTraining() also does this so not necessary to call both. 
        :return: 
        '''
        threshTemp = self.threshold / np.max(self.threshold)
        threshTemp = ['{0:.4f}'.format(i) for i in threshTemp]

        threshTemp2 = self.threshold2 / np.max(self.threshold2)
        threshTemp2 = ['{0:.4f}'.format(i) for i in threshTemp2]

        logger.debug("Scale Factor" + ",".join(threshTemp))
        logger.debug("Scale GFactor" + ",".join(threshTemp2))

        self.threshold = np.ones(self.dataset.classes, dtype=np.float64)
        self.threshold2 = np.ones(self.dataset.classes, dtype=np.float64)

    def setup_training(self):
        self.resetThresh()

        for param_group in self.optimizer.param_groups:
            logger.debug("Setting LR to %0.2f", self.args.lr)
            param_group['lr'] = self.args.lr
            self.current_lr = self.args.lr
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
                logger.warning("JM model initializes as the model")

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

    def getModel(self):
        myModel = model.ModelFactory.get_model(self.args.model_type, self.args.dataset)

        if self.args.cuda:
            myModel.cuda()

        optimizer = torch.optim.SGD(myModel.parameters(), self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.decay, nesterov=True)
        myModel.eval()

        self.current_lr = self.args.lr

        self.model_single = myModel
        self.optimizer_single = optimizer

    ################ Function for Norm Jacobian calculation ################
    def compute_normalized_jacobian(self, data, use_fixed_model=True, use_model_jm=False,
                                    is_norm=True, is_calc_from_embedded=True):

        if is_calc_from_embedded:
            jacobian = self.compute_jacobian_from_embedded(data, use_fixed_model, use_model_jm)

            if is_norm:
                a = jacobian.detach()
                a = a.view(10, self.args.batch_size, -1)
                b = torch.norm(a, dim=(0, 2))
                b = b.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                jacobian_norm = jacobian.div(b.expand_as(jacobian))

                return jacobian_norm

            else:
                return jacobian
        else:
            jacobian = self.compute_jacobian(data, use_fixed_model, use_model_jm)

            if is_norm:
                a = jacobian.detach()
                a = a.view(len(self.older_classes), self.args.batch_size, -1)
                b = torch.norm(a, dim=(0, 2))
                b = b.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                jacobian_norm = jacobian.div(b.expand_as(jacobian))

                return jacobian_norm

            else:
                return jacobian

    ################ Function for Jacobian calculation ################
    def compute_jacobian(self, data, use_fixed_model=True, use_model_jm=False):

        inputs = Variable(data, requires_grad=True)

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

        num_classes = output.size()[1]

        jacobian_list = []
        grad_output = torch.zeros(*output.size())

        if inputs.is_cuda:
            grad_output = grad_output.cuda()

        for i in self.older_classes:
            zero_gradients(inputs)

            grad_output_curr = grad_output.clone()
            grad_output_curr[:, i] = 1
            jacobian_list.append(torch.autograd.grad(outputs=output,
                                                     inputs=inputs,
                                                     grad_outputs=grad_output_curr,
                                                     only_inputs=True,
                                                     retain_graph=True,
                                                     create_graph=not use_fixed_model)[0])

        jacobian = torch.stack(jacobian_list, dim=0)

        return jacobian

    ################ Function for Jacobian calculation ################
    def compute_jacobian_from_embedded(self, data, use_fixed_model=True, use_model_jm=False):

        inputs = Variable(data, requires_grad=True)

        if not use_model_jm:
            if use_fixed_model:
                output, embedded = self.model_fixed.forward(inputs, embedding_space=True)
            else:
                output, embedded = self.model.forward(inputs, embedding_space=True)
        else:
            if use_fixed_model:
                output, embedded = self.model_fixed_jm.forward(inputs, embedding_space=True)
            else:
                output, embedded = self.model_jm.forward(inputs, embedding_space=True)

        random_normal_mat = torch.randn(embedded.size()[0], 10, embedded.size()[-1])

        if self.args.cuda:
            random_normal_mat = random_normal_mat.cuda()

        embedded_projection = matmul(random_normal_mat, embedded)

        jacobian_list = []
        grad_output = torch.zeros(*embedded_projection.size())

        if inputs.is_cuda:
            grad_output = grad_output.cuda()

        for i in range(embedded_projection.size()[0]):
            zero_gradients(inputs)

            grad_output_curr = grad_output.clone()
            grad_output_curr[i, :] = 1
            jacobian_list.append(torch.autograd.grad(outputs=embedded_projection,
                                                     inputs=inputs,
                                                     grad_outputs=grad_output_curr,
                                                     only_inputs=True,
                                                     retain_graph=True,
                                                     create_graph=not use_fixed_model)[0])

        jacobian = torch.stack(jacobian_list, dim=0)

        return jacobian

    def do_backward_patch(self):
        return True
        # return (len(self.older_classes) < 2)

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

    def train(self, epoch, is_jacobian_matching=True, use_model_jm=False):

        self.model.train()
        if use_model_jm:
            self.model_jm.train()

        logger.info("Epoch %d", epoch)
        torch.manual_seed(self.seed)

        for data, y, target in tqdm(self.train_data_iterator):

            if self.args.batch_size != target.__len__():
                print('Skip due to small target len')
                continue

            if 0 == epoch:
                try:
                    bin_count += target.bincount()
                except NameError:
                    bin_count = target.bincount()

            if self.args.cuda:
                data, target, y = data.cuda(), target.cuda(), y.cuda()

            old_classes_indices = (target * 0).int()

            for elem in range(0, self.args.unstructured_size):
                old_classes_indices = old_classes_indices + (target == elem).int()

            # old_classes_indices = torch.squeeze(torch.nonzero((old_classes_indices > 0)).long())
            new_classes_indices = torch.squeeze(torch.nonzero((old_classes_indices == 0)).long())

            self.optimizer.zero_grad()
            if use_model_jm:
                self.optimizer_jm.zero_grad()

            # Use only new classes for normal classification_loss:
            target_normal_loss = target[new_classes_indices]
            data_normal_loss = data[new_classes_indices]

            # Use all of the data for distillation loss:
            # target_distillation_loss = y.float()
            data_distillation_loss = data

            # Create y_onehot tensor for normal classification_loss
            try:
                y_onehot = torch.FloatTensor(len(target_normal_loss), self.dataset.classes)
            except:
                print('Fatal!')
                continue

            if self.args.cuda:
                y_onehot = y_onehot.cuda()

            y_onehot.zero_()
            target_normal_loss.unsqueeze_(1)
            y_onehot.scatter_(1, target_normal_loss, 1)

            self.threshold += np.sum(y_onehot.cpu().numpy(), 0)

            output = self.model(Variable(data_normal_loss))
            classification_loss = F.kl_div(output, Variable(y_onehot))

            if use_model_jm:
                output_jm = self.model_jm(Variable(data_normal_loss))
                classification_loss_jm = F.kl_div(output_jm, Variable(y_onehot))

            myT = self.args.T

            if self.args.no_distill:
                pass

            elif len(self.older_classes) > 0:
                # Get softened labels of the model from a previous version of the model.

                if not is_jacobian_matching or use_model_jm:
                    pred2 = self.model_fixed(Variable(data_distillation_loss), T=myT, labels=True).data

                    # Softened output of the model
                    if myT > 1:
                        output2 = self.model(Variable(data_distillation_loss), T=myT)
                    else:
                        output2 = output
                        pred2 = pred2[new_classes_indices, :] #My addition due to fail (Einav)

                    # distillation_loss = F.kl_div(output2[:, self.older_classes], Variable(pred2[:, self.older_classes]))
                    distillation_loss = F.kl_div(output2, Variable(pred2))
                    if self.do_backward_patch():
                        distillation_loss.backward(retain_graph=True)
                    self.threshold += (np.sum(pred2.cpu().numpy(), 0)) * (myT * myT) * self.args.alpha

                if is_jacobian_matching or use_model_jm:
                    # #TODO: Change cpu threshold for jacobian_matching_loss (Einav)

                    jacobian = self.compute_normalized_jacobian(data, use_fixed_model=False,
                                                                use_model_jm=use_model_jm,
                                                                is_norm=self.args.norm_jacobian,
                                                                is_calc_from_embedded=True)

                    jacobian_model_fixed = self.compute_normalized_jacobian(data, use_fixed_model=True,
                                                                            use_model_jm=use_model_jm,
                                                                            is_norm=self.args.norm_jacobian,
                                                                            is_calc_from_embedded=True)

                    jacobian_matching_loss = self.decay_jm*torch.norm(jacobian - jacobian_model_fixed)

                    if self.do_backward_patch():
                        jacobian_matching_loss.backward(retain_graph=True)

                    # if DEBUG and 0 == epoch:
                    #     self.plot_3d(jacobian.detach())

                self.scale_gradient_by_square_of_T(myT, use_model_jm)

            if self.do_backward_patch():
                if len(self.older_classes) == 0 or not self.args.no_nl:
                    classification_loss.backward()
                    if use_model_jm:
                        classification_loss_jm.backward()

            for param in self.model.named_parameters():
                if "fc.weight" in param[0]:
                    self.threshold2 *= 0.99
                    self.threshold2 += np.sum(np.abs(param[1].grad.data.cpu().numpy()), 1)

            self.optimizer.step()
            if use_model_jm:
                self.optimizer_jm.step()

        if 0 == epoch:
            bin_count_norm = bin_count.float() / bin_count.float().sum()
            logger.debug("bin_count: " + str(bin_count.tolist()))
            logger.debug("bin_count_norm: " + str(bin_count_norm.tolist()))

        if self.args.no_nl:
            self.threshold[len(self.older_classes):len(self.threshold)] = np.max(self.threshold)
            self.threshold2[len(self.older_classes):len(self.threshold2)] = np.max(self.threshold2)

        else:
            self.threshold[0:self.args.unstructured_size] = np.max(self.threshold)
            self.threshold2[0:self.args.unstructured_size] = np.max(self.threshold2)

            self.threshold[self.args.unstructured_size +
                           len(self.older_classes) +
                           self.args.step_size: len(self.threshold)] = np.max(self.threshold)
            self.threshold2[self.args.unstructured_size +
                            len(self.older_classes) +
                            self.args.step_size: len(self.threshold2)] = np.max(self.threshold2)

        if logger is not None and epoch % 10 == (10 - 1):
            logger.debug("*********CURRENT EPOCH********** : %d", epoch)
            logger.debug("Classification Loss: %0.4f", classification_loss)
            if not self.args.no_distill and len(self.older_classes) > 0:
                logger.debug("Distillation Loss: %0.4f", distillation_loss)
                logger.debug("Jacobian Matching Loss: %0.4f", jacobian_matching_loss)

    def addModel(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        logger.debug("Total Models %d", len(self.models))

    def trainSingle(self, epoch, classGroup):

        for temp in range(0, len(self.args.schedule)):

            if self.args.schedule[temp] == epoch:

                for param_group in self.optimizer_single.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    logger.debug("Changing learning rate from %0.2f to %0.2f", self.current_lr,
                                 self.current_lr * self.args.gammas[temp])
                    self.current_lr *= self.args.gammas[temp]

        self.model_single.train()

        for batch_idx, (data, y, target) in enumerate(self.train_data_iterator):

            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
                y = y.cuda()

            oldClassesIndices = (target * 0).int()

            for elem in range(0, self.args.unstructured_size + classGroup):
                oldClassesIndices = oldClassesIndices + (target == elem).int()

            new_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices == 0)).long())

            if len(new_classes_indices) > 0:
                self.optimizer_single.zero_grad()

                target_normal_loss = y[new_classes_indices]
                data_normal_loss = data[new_classes_indices]

                y_onehot = target_normal_loss.float()

                output = self.model_single(Variable(data_normal_loss))
                loss = F.kl_div(output, Variable(y_onehot))

                loss.backward()

                self.optimizer_single.step()

    def storeDistillation(self, epoch, classGroup):

        self.train_data_iterator.dataset.getIndexElem(True)

        for batch_idx, (data, y, target) in enumerate(self.train_data_iterator):

            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
                y = y.cuda()

            oldClassesIndices = (target * 0).int()
            for elem in range(0, self.args.unstructured_size + classGroup):
                oldClassesIndices = oldClassesIndices + (target == elem).int()

            new_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices == 0)).long())

            indices = y[new_classes_indices]
            data_normal_loss = data[new_classes_indices]

            output = self.model_single(Variable(data_normal_loss), labels=True, T=self.args.T)
            output = output.data.cpu().numpy()
            self.train_data_iterator.dataset.labels[indices] = output
            # print (self.train_data_iterator.dataset.labels[indices[0]], "SUM", np.sum(self.train_data_iterator.dataset.labels[indices[0]]))

        self.train_data_iterator.dataset.getIndexElem(False)

    def save_models(self,  file_name):
        torch.save(self.model.state_dict(), file_name + '.pth')
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

