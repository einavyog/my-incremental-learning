''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from __future__ import print_function

import argparse
import logging

import torch
import torch.utils.data as td
import sys
from torch import nn

import data_handler
import experiment as ex
import model
import plotter as plt
import trainer
import copy
import seaborn as sns
from matplotlib import pyplot

import utils.Colorer

import os
from subprocess import check_output

# os.environ["CUDA_VISIBLE_DEVICES"]="2"
sys.version
print("PyTorch version: ")
print(torch.__version__)
print("CUDA Version: ")
print(torch.version.cuda)
print("cuDNN version is: ")
print(torch.backends.cudnn.version())

from subprocess import check_output

logger = logging.getLogger('iCARL')

parser = argparse.ArgumentParser(description='iCarl2.0')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=2.0, metavar='LR',
                    help='learning rate (default: 2.0). Note that lr is decayed by args.gamma parameter args.schedule ')
parser.add_argument('--schedule', type=int, nargs='+', default=[45, 60, 68],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--random-init', action='store_true', default=False,
                    help='Initialize model for next increment using previous weights if false and random weights otherwise')
parser.add_argument('--no-distill', action='store_true', default=False,
                    help='disable distillation loss and only uses the cross entropy loss. See "Distilling Knowledge in Neural Networks" by Hinton et.al for details')
parser.add_argument('--no-random', action='store_true', default=False,
                    help='Disable random shuffling of classes')
parser.add_argument('--no-herding', action='store_true', default=False,
                    help='Disable herding algorithm and do random instance selection instead')
parser.add_argument('--seeds', type=int, nargs='+', default=[23423],
                    help='Seeds values to be used; seed introduces randomness by changing order of classes')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type', default="resnet32",
                    help='model type to be used. Example : resnet32, resnet20, test')
parser.add_argument('--name', default="noname",
                    help='Name of the experiment')
parser.add_argument('--outputDir', default="./results/",
                    help='Directory to store the results; a new folder "DDMMYYYY" will be created '
                         'in the specified directory to save the results.')
parser.add_argument('--upsampling', action='store_true', default=False,
                    help='Do not do upsampling.')
parser.add_argument('--unstructured-size', type=int, default=0,
                    help='Leftover parameter of an unreported experiment; leave it at 0')
parser.add_argument('--alphas', type=float, nargs='+', default=[1.0],
                    help='Weight given to new classes vs old classes in the loss; high value of alpha will increase perfomance on new classes at the expense of older classes. Dynamic threshold moving makes the system more robust to changes in this parameter')
parser.add_argument('--decay', type=float, default=0.00005, help='Weight decay (L2 penalty).')
parser.add_argument('--step-size', type=int, default=10, help='How many classes to add in each increment')
parser.add_argument('--T', type=float, default=1, help='Tempreture used for softening the targets')
parser.add_argument('--memory-budgets', type=int, nargs='+', default=[2000],
                    help='How many images can we store at max. 0 will result in fine-tuning')
parser.add_argument('--epochs-class', type=int, default=70, help='Number of epochs for each increment')
parser.add_argument('--pca_dim', type=int, default=10, help='Number of dimensions in PCA projection')
parser.add_argument('--dataset', default="CIFAR100", help='Dataset to be used; example CIFAR, MNIST')
parser.add_argument('--lwf', action='store_true', default=False,
                    help='Use learning without forgetting. Ignores memory-budget '
                         '("Learning with Forgetting," Zhizhong Li, Derek Hoiem)')
parser.add_argument('--no-nl', action='store_true', default=False,
                    help='No Normal Loss. Only uses the distillation loss to train the new model on old classes (Normal loss is used for new classes however')
parser.add_argument('--pretrained_model', default=None,
                    help='Path to model weights')
parser.add_argument('--pretrained_model_jm', default=None,
                    help='Path to model weights for jacobian matching model')
parser.add_argument('--jm_decay', type=float, default=0.001, help='Jacobian Matching decay (L2 penalty).')
parser.add_argument('--activation_decay', type=float, default=0.0005, help='Jacobian Matching decay (L2 penalty).')
parser.add_argument('--norm_jacobian', action='store_true', default=False,
                    help='Use normed Jacobian for Jacobiam matchon. Relevanty only when using --jacobian_matching')
parser.add_argument('--jacobian_matching', action='store_true', default=False)
parser.add_argument('--no_projection', action='store_true', default=False)
parser.add_argument('--no_jm_classification', action='store_true', default=False)
parser.add_argument('--no_jm_loss', action='store_true', default=False)
parser.add_argument('--use_activation_matching', action='store_true', default=False)
parser.add_argument('--use_pca', action='store_true', default=False)
parser.add_argument('--use_distillation', action='store_true', default=False)
parser.add_argument('--no_bn', action='store_true', default=False)
parser.add_argument('--match_one_layer', action='store_true', default=False)
parser.add_argument('--project_outputs', action='store_true', default=False)
parser.add_argument('--projection_dim', type=int, default=10,
                    help='The size of the random matrix prior to the Jacobian Matching')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

IS_RUN_LOCAL = False
ips = check_output(['hostname', '--all-ip-addresses'])
if ips == b'132.66.50.93 \n':
    IS_RUN_LOCAL = True
    print('running local')

args.is_run_local = IS_RUN_LOCAL
dataset = data_handler.DatasetFactory.get_dataset(args.dataset)

# Checks to make sure parameters are sane
if args.step_size < 2:
    print("Step size of 1 will result in no learning;")
    assert False

# Plotting the line diagrams of all the possible cases
y_total, y1_total, train_y_total, y_total_jm, y1_total_jm, train_y_total_jm = ([] for i in range(6))

# Run an experiment corresponding to every seed value
for seed in args.seeds:
    # Run an experiment corresponding to every alpha value
    for at in args.alphas:
        args.alpha = at
        # Run an experiment corresponding to every memory budget
        for m in args.memory_budgets:
            args.memory_budget = m
            # In LwF, memory_budget is 0 (See the paper "Learning without Forgetting" for details).
            if args.lwf:
                args.memory_budget = 0

            # Fix the seed.
            args.seed = seed
            torch.manual_seed(seed)
            if args.cuda:
                torch.cuda.manual_seed(seed)

            # Loader used for training data
            train_dataset_loader = data_handler.IncrementalLoader(dataset.train_data.train_data,
                                                                  dataset.train_data.train_labels,
                                                                  dataset.labels_per_class_train,
                                                                  dataset.classes, [0, 1],
                                                                  transform=dataset.train_transform,
                                                                  cuda=args.cuda, oversampling=not args.upsampling)
            # Special loader use to compute ideal NMC; i.e, NMC that using all the data points to compute the mean embedding
            train_dataset_loader_nmc = data_handler.IncrementalLoader(dataset.train_data.train_data,
                                                                      dataset.train_data.train_labels,
                                                                      dataset.labels_per_class_train,
                                                                      dataset.classes, [0, 1],
                                                                      transform=dataset.train_transform,
                                                                      cuda=args.cuda, oversampling=not args.upsampling)
            # Loader for test data.
            test_dataset_loader = data_handler.IncrementalLoader(dataset.test_data.test_data,
                                                                 dataset.test_data.test_labels,
                                                                 dataset.labels_per_class_test, dataset.classes,
                                                                 [0, 1], transform=dataset.test_transform, cuda=args.cuda)

            kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

            # Iterator to iterate over training data.
            train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                                         batch_size=args.batch_size, shuffle=True, **kwargs)
            # Iterator to iterate over all training data (Equivalent to memory-budget = infitie
            train_iterator_nmc = torch.utils.data.DataLoader(train_dataset_loader_nmc,
                                                             batch_size=args.batch_size, shuffle=True, **kwargs)
            # Iterator to iterate over test data
            test_iterator = torch.utils.data.DataLoader(
                test_dataset_loader,
                batch_size=args.batch_size, shuffle=True, **kwargs)

            # Get the required model
            myModel = model.ModelFactory.get_model(args.model_type, args.dataset)
            if args.cuda:
                myModel = nn.DataParallel(myModel)
                myModel.cuda()

            if args.jacobian_matching:
                # Get the required model
                myModel_jm = model.ModelFactory.get_model(args.model_type, args.dataset)
                if args.cuda:
                    myModel_jm = nn.DataParallel(myModel_jm)
                    myModel_jm.cuda()

                myModel_jm.load_state_dict(copy.deepcopy(myModel.state_dict()))

            else:
                myModel_jm = None

            # Define an experiment.
            my_experiment = ex.experiment(args.name, args, output_dir=args.outputDir)

            # Adding support for logging. A .log is generated with all the logs. Logs are also stored in a temp file one directory
            # before the code repository
            logger = logging.getLogger('iCARL')
            logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler(my_experiment.path + ".log")
            fh.setLevel(logging.DEBUG)

            fh2 = logging.FileHandler("../temp.log")
            fh2.setLevel(logging.DEBUG)

            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            fh2.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(fh2)
            logger.addHandler(ch)

            logger.info("Input Args:")
            for arg in vars(args):
                logger.info("%s: %s", arg, str(getattr(args, arg)))

            # Define the optimizer used in the experiment
            optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                                        weight_decay=args.decay, nesterov=True)

            if args.jacobian_matching:
                # Define the optimizer used in the experiment
                optimizer_jm = torch.optim.SGD(myModel_jm.parameters(), args.lr, momentum=args.momentum,
                                               weight_decay=args.decay, nesterov=True)
            else:
                optimizer_jm = None

            # Trainer object used for training
            my_trainer = trainer.Trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer,
                                         train_iterator_nmc, myModel_jm, optimizer_jm)

            # Parameters for storing the results
            x, y, y1, train_y, higher_y, y_scaled, y_grad_scaled, nmc_ideal_cum = ([] for i in range(8))
            y_jm, y1_jm, train_y_jm, y_scaled_jm, y_grad_scaled_jm, nmc_ideal_cum_jm = ([] for i in range(6))

            # Initilize the evaluators used to measure the performance of the system.
            nmc = trainer.EvaluatorFactory.get_evaluator("nmc", args.cuda)
            nmc_ideal = trainer.EvaluatorFactory.get_evaluator("nmc", args.cuda)
            t_classifier = trainer.EvaluatorFactory.get_evaluator("trainedClassifier", args.cuda)

            # Initilize the evaluators used to measure the performance of the system.
            nmc_jm = trainer.EvaluatorFactory.get_evaluator("nmc", args.cuda)
            nmc_ideal_jm = trainer.EvaluatorFactory.get_evaluator("nmc", args.cuda)
            t_classifier_jm = trainer.EvaluatorFactory.get_evaluator("trainedClassifier", args.cuda)

            # Loop that incrementally adds more and more classes
            for class_group in range(0, dataset.classes, args.step_size):
                print("SEED:", seed, "MEMORY_BUDGET:", m, "CLASS_GROUP:", class_group)
                # Add new classes to the train, train_nmc, and test iterator
                my_trainer.increment_classes(class_group)
                my_trainer.update_frozen_model(args.jacobian_matching)
                epoch = 0

                # Running epochs_class epochs
                train_error_all_epochs, test_error_all_epochs, epochs_list = ([] for i in range(3))
                train_error_all_epochs_jm, test_error_all_epochs_jm = ([] for i in range(2))

                for epoch in range(0, args.epochs_class):
                    my_trainer.update_lr(epoch, use_jm=args.jacobian_matching)
                    my_trainer.train(epoch, use_model_jm=args.jacobian_matching)

                    # print(my_trainer.threshold)
                    if epoch % args.log_interval == (args.log_interval - 1):
                        train_error = t_classifier.evaluate(my_trainer.model, train_iterator)
                        test_error = t_classifier.evaluate(my_trainer.model, test_iterator)
                        logger.debug("*********CURRENT EPOCH********** : %d", epoch)
                        logger.debug("Train Classifier: %0.4f", train_error)
                        logger.debug("Test Classifier: %0.4f", test_error)
                        train_error_all_epochs.append(train_error)
                        test_error_all_epochs.append(test_error)
                        epochs_list.append(epoch)

                        if args.jacobian_matching:
                            train_error_jm = t_classifier.evaluate(my_trainer.model_jm, train_iterator)
                            test_error_jm = t_classifier.evaluate(my_trainer.model_jm, test_iterator)
                            logger.debug("Train Classifier JM: %0.4f", train_error_jm)
                            logger.debug("Test Classifier JM: %0.4f", test_error_jm)

                            train_error_all_epochs_jm.append(train_error_jm)
                            test_error_all_epochs_jm.append(test_error_jm)

                # Evaluate the learned classifier
                img = None

                logger.info("Test Classifier Final: %0.2f", t_classifier.evaluate(my_trainer.model, test_iterator))
                logger.info("Test Classifier Final Scaled: %0.2f",
                            t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.dynamic_threshold, False,
                                                  my_trainer.older_classes, args.step_size))
                logger.info("Test Classifier Final Grad Scaled: %0.2f",
                            t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.gradient_threshold_unreported_experiment, False,
                                                  my_trainer.older_classes, args.step_size))

                higher_y.append(t_classifier.evaluate(my_trainer.model, test_iterator, higher=True))

                y_grad_scaled.append(
                    t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.gradient_threshold_unreported_experiment, False,
                                          my_trainer.older_classes, args.step_size))
                y_scaled.append(t_classifier.evaluate(my_trainer.model, test_iterator, my_trainer.dynamic_threshold, False,
                                                      my_trainer.older_classes, args.step_size))
                y1.append(t_classifier.evaluate(my_trainer.model, test_iterator))

                # Update means using the train iterator; this is iCaRL case
                nmc.update_means(my_trainer.model, train_iterator, dataset.classes)
                # Update mean using all the data. This is equivalent to memory_budget = infinity
                nmc_ideal.update_means(my_trainer.model, train_iterator_nmc, dataset.classes)
                # Compute the the nmc based classification results
                tempTrain = t_classifier.evaluate(my_trainer.model, train_iterator)
                train_y.append(tempTrain)

                if args.use_pca:
                    my_trainer.update_pca(args.pca_dim, use_jm=args.jacobian_matching)

                testY1 = nmc.evaluate(my_trainer.model, test_iterator, step_size=args.step_size, kMean=True)
                testY = nmc.evaluate(my_trainer.model, test_iterator)
                testY_ideal = nmc_ideal.evaluate(my_trainer.model, test_iterator)
                y.append(testY)
                nmc_ideal_cum.append(testY_ideal)

                if args.jacobian_matching:
                    y_grad_scaled_jm.append(t_classifier_jm.evaluate(my_trainer.model_jm,
                                                                     test_iterator, my_trainer.gradient_threshold_unreported_experiment, False,
                                                                     my_trainer.older_classes, args.step_size))
                    y_scaled_jm.append(t_classifier_jm.evaluate(my_trainer.model_jm,
                                                                test_iterator, my_trainer.dynamic_threshold, False,
                                                                my_trainer.older_classes, args.step_size))

                    y1_jm.append(t_classifier_jm.evaluate(my_trainer.model_jm, test_iterator))

                    # Update means using the train iterator; this is iCaRL case #TODO: Einav. check issue
                    nmc_jm.update_means(my_trainer.model_jm, train_iterator, dataset.classes)
                    # Update mean using all the data. This is equivalent to memory_budget = infinity
                    nmc_ideal_jm.update_means(my_trainer.model_jm, train_iterator_nmc, dataset.classes)

                    # Compute the the nmc based classification results
                    train_y_jm.append(t_classifier_jm.evaluate(my_trainer.model_jm, train_iterator))

                    y_jm.append(nmc_jm.evaluate(my_trainer.model_jm, test_iterator))
                    nmc_ideal_cum_jm.append(nmc_ideal_jm.evaluate(my_trainer.model_jm, test_iterator))

                # Compute confusion matrices of all three cases (Learned classifier, iCaRL, and ideal NMC)
                tcMatrix = t_classifier.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes)
                tcMatrix_scaled = t_classifier.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes,
                                                                    my_trainer.dynamic_threshold, my_trainer.older_classes,
                                                                    args.step_size)
                tcMatrix_grad_scaled = t_classifier.get_confusion_matrix(my_trainer.model, test_iterator,
                                                                         dataset.classes,
                                                                         my_trainer.gradient_threshold_unreported_experiment,
                                                                         my_trainer.older_classes,
                                                                         args.step_size)
                nmcMatrix = nmc.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes)
                nmcMatrixIdeal = nmc_ideal.get_confusion_matrix(my_trainer.model, test_iterator, dataset.classes)
                tcMatrix_scaled_binning = t_classifier.get_confusion_matrix(my_trainer.model, test_iterator,
                                                                            dataset.classes,
                                                                            my_trainer.dynamic_threshold,
                                                                            my_trainer.older_classes,
                                                                            args.step_size, True)

                if args.jacobian_matching:
                    # Compute confusion matrices of all three cases (Learned classifier, iCaRL, and ideal NMC)
                    tcMatrix_jm = t_classifier_jm.get_confusion_matrix(my_trainer.model_jm, test_iterator, dataset.classes)
                    tcMatrix_scaled_jm = t_classifier_jm.get_confusion_matrix(my_trainer.model_jm, test_iterator, dataset.classes,
                                                                        my_trainer.dynamic_threshold, my_trainer.older_classes,
                                                                        args.step_size)
                    tcMatrix_grad_scaled_jm = t_classifier_jm.get_confusion_matrix(my_trainer.model_jm, test_iterator,
                                                                             dataset.classes,
                                                                             my_trainer.gradient_threshold_unreported_experiment,
                                                                             my_trainer.older_classes,
                                                                             args.step_size)
                    nmcMatrix_jm = nmc_jm.get_confusion_matrix(my_trainer.model_jm, test_iterator, dataset.classes)
                    nmcMatrixIdeal_jm = nmc_ideal_jm.get_confusion_matrix(my_trainer.model_jm, test_iterator, dataset.classes)
                    tcMatrix_scaled_binning_jm = t_classifier_jm.get_confusion_matrix(my_trainer.model_jm, test_iterator,
                                                                                dataset.classes,
                                                                                my_trainer.dynamic_threshold,
                                                                                my_trainer.older_classes,
                                                                                args.step_size, True)

                my_trainer.setup_training(use_jm=True)

                # Store the resutls in the my_experiment object; this object should contain all the information required to reproduce the results.
                x.append(class_group + args.step_size)

                my_experiment.results["NMC"] = [x, [float(p) for p in y]]
                my_experiment.results["Trained Classifier"] = [x, [float(p) for p in y1]]
                my_experiment.results["Trained Classifier Scaled"] = [x, [float(p) for p in y_scaled]]
                my_experiment.results["Trained Classifier Grad Scaled"] = [x, [float(p) for p in y_grad_scaled]]
                my_experiment.results["Train Error Classifier"] = [x, [float(p) for p in train_y]]
                my_experiment.results["Ideal NMC"] = [x, [float(p) for p in nmc_ideal_cum]]

                if args.jacobian_matching:
                    my_experiment.results["NMC JM"] = [x, [float(p) for p in y_jm]]
                    my_experiment.results["Trained Classifier JM"] = [x, [float(p) for p in y1_jm]]
                    my_experiment.results["Trained Classifier Scaled JM"] = [x, [float(p) for p in y_scaled_jm]]
                    my_experiment.results["Trained Classifier Grad Scaled JM"] = [x, [float(p) for p in y_grad_scaled_jm]]
                    my_experiment.results["Train Error Classifier JM"] = [x, [float(p) for p in train_y_jm]]
                    my_experiment.results["Ideal NMC JM"] = [x, [float(p) for p in nmc_ideal_cum_jm]]

                my_experiment.store_json()

                # Finally, plotting the results;
                my_plotter = plt.Plotter()

                # Plotting the confusion matrices
                my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                      my_experiment.path + "_tc_matrix", tcMatrix)
                my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                      my_experiment.path + "_tc_matrix_scaled", tcMatrix_scaled)
                my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                      my_experiment.path + "_tc_matrix_scaled_binning", tcMatrix_scaled_binning)
                my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                      my_experiment.path + "_nmc_matrix",
                                      nmcMatrix)
                my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                      my_experiment.path + "_nmc_matrix_ideal",
                                      nmcMatrixIdeal)
                if args.jacobian_matching:
                    # Plotting the confusion matrices
                    my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                          my_experiment.path + "_tc_matrix_jm", tcMatrix_jm)
                    my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                          my_experiment.path + "_tc_matrix_scaled_jm", tcMatrix_scaled_jm)
                    my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                          my_experiment.path + "_tc_matrix_scaled_binning_jm", tcMatrix_scaled_binning_jm)
                    my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                          my_experiment.path + "_nmc_matrix_jm",
                                          nmcMatrix_jm)
                    my_plotter.plotMatrix(int(class_group / args.step_size) * args.epochs_class + epoch,
                                          my_experiment.path + "_nmc_matrix_ideal_jm",
                                          nmcMatrixIdeal_jm)

                # Plotting the line diagrams of all the possible cases
                my_plotter.plot(x, y, title=args.name, legend="NMC")
                # my_plotter.plot(x, higher_y, title=args.name, legend="Higher Model")
                my_plotter.plot(x, y_scaled, title=args.name, legend="Trained Classifier Scaled")
                # my_plotter.plot(x, y_grad_scaled, title=args.name, legend="Trained Classifier Grad Scaled")
                #my_plotter.plot(x, nmc_ideal_cum, title=args.name, legend="Ideal NMC")
                my_plotter.plot(x, y1, title=args.name, legend="Trained Classifier")
                my_plotter.plot(x, train_y, title=args.name, legend="Trained Classifier Train Set")

                if args.jacobian_matching:
                    my_plotter.plot(x, y_jm, title=args.name, legend="NMC jm")
                    # my_plotter.plot(x, higher_y, title=args.name, legend="Higher Model")
                    my_plotter.plot(x, y_scaled_jm, title=args.name, legend="Trained Classifier Scaled jm")
                    # my_plotter.plot(x, y_grad_scaled_jm, title=args.name, legend="Trained Classifier Grad Scaled jm")
                    # my_plotter.plot(x, nmc_ideal_cum_jm, title=args.name, legend="Ideal NMC jm")
                    my_plotter.plot(x, y1_jm, title=args.name, legend="Trained Classifier jm")
                    my_plotter.plot(x, train_y_jm, title=args.name, legend="Trained Classifier Train Set jm")


                # Saving the line plot
                my_plotter.save_fig(my_experiment.path, dataset.classes)
                my_trainer.save_models(my_experiment.path, use_model_jm=args.jacobian_matching)

                pyplot.plot(epochs_list, train_error_all_epochs, label="training", marker='o', linestyle='dashed')
                pyplot.plot(epochs_list, test_error_all_epochs, label="testing", marker='o', linestyle='dashed')
                pyplot.plot(epochs_list, train_error_all_epochs_jm, label="training jm", marker='o', linestyle='dashed')
                pyplot.plot(epochs_list, test_error_all_epochs_jm, label="testing jm", marker='o', linestyle='dashed')
                pyplot.xlabel('Epoch')
                pyplot.ylabel('Accuracy')
                pyplot.title('Accuracy Class Group ' + str(class_group))
                pyplot.legend()
                pyplot.grid(True)
                pyplot.savefig(my_experiment.path + '_accuracy_per_epoch_class_group' + str(class_group))

            y_total.append(y)
            y_total_jm.append(y_jm)
            y1_total_jm.append(y1_jm)
            train_y_total_jm.append(train_y_jm)
            y1_total.append(y1)
            train_y_total.append(train_y)

# Plot avarage over all runs:
ncols = len(y_total[0])
nrows = len(y_total)

y_total_avg, y_total_jm_avg, y1_total_jm_avg, train_y_total_jm_avg, y1_total_avg, train_y_total_avg = \
    (ncols*[0] for i in range(6))

nelem = float(nrows)
col = 0
for col in range(ncols):
    for row in range(nrows):
        y_total_avg[col] += y_total[row][col]
        y1_total_avg[col] += y1_total[row][col]
        train_y_total_avg[col] += train_y_total[row][col]
        y_total_jm_avg[col] += y_total_jm[row][col]
        y1_total_jm_avg[col] += y1_total_jm[row][col]
        train_y_total_jm_avg[col] += train_y_total_jm[row][col]

    y_total_avg[col] /= nelem
    y_total_jm_avg[col] /= nelem
    y1_total_jm_avg[col] /= nelem
    train_y_total_jm_avg[col] /= nelem
    y1_total_avg[col] /= nelem
    train_y_total_avg[col] /= nelem

my_plotter_total = plt.Plotter()

# Plotting the line diagrams of all the possible cases
my_plotter_total.plot(x, y_total_avg, title=args.name, legend="NMC")
my_plotter_total.plot(x, y_total_jm_avg, title=args.name, legend="NMC JM")
my_plotter_total.plot(x, y1_total_avg, title=args.name, legend="Trained Classifier")
my_plotter_total.plot(x, y1_total_jm_avg, title=args.name, legend="Trained Classifier JM")
my_plotter_total.plot(x, train_y_total_avg, title=args.name, legend="Trained Classifier Train Set")
my_plotter_total.plot(x, train_y_total_jm_avg, title=args.name, legend="Trained Classifier Train Set JM")

# Saving the line plot
my_plotter_total.save_fig(my_experiment.path + "_avg_over_all_seeds",
                          dataset.classes + 1, title='Avg over ' + str(len(args.seeds)) + ' epochs')

# import numpy as np
# import pandas as pd
#
# y_total_df = pd.DataFrame(columns=['Number Of Classes', 'Accuracy'])
# a = list(map(list, zip(*y_total)))
# col = 0
# for class_group in range(0, dataset.classes, args.step_size):
#     curr_class = class_group + args.step_size
#     print(curr_class)
#     row = 0
#     for seed in args.seeds:
#         tmp = np.transpose([np.array(y_total)[:, col].astype(int), [curr_class, curr_class]])
#         y_total_df = y_total_df.append(pd.DataFrame(tmp, columns=['Number Of Classes', 'Accuracy']), ignore_index=True)
#         row += 1
#     col += 1
#
# print(y_total_df)
# print(y_total)
#
# y_total_df = y_total_df.astype(int)
# a = sns.lineplot(x='Number Of Classes', y='Accuracy', data=y_total_df, legend='full').get_figure()
# a.savefig(my_experiment.path + "_avg_over_all_seeds.pdf")
