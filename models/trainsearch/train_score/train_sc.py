
import models.auxmodal.scheduler as aux
from models.trainsearch.darts.utils import count_parameters, save, save_pickle
from torch.autograd import Variable
import torch.nn as nn
##################from torch.autograd import Variable#
import torch
import copy
from sklearn.metrics import mean_squared_error #mean_squared_error
from tqdm import tqdm
import os

def criterion(input,target): ##############추가
    loss = nn.MSELoss()
    loss1 = loss(input,target)#######################
    return loss1


def train_track_mse(  model, architect,  criterion, optimizer, scheduler,  dataloaders,  dataset_sizes, device, num_epochs,  parallel, logger, plotter, args,
                           init_mse=1, th_mse2=0.0001,   status='search'): #mse_type='weighted',

    best_genotype = None
    least_mse = init_mse
    best_epoch = 0

    best_test_genotype = None
    least_test_mse = init_mse
    best_test_epoch = 0

    failsafe = True

    cont_overloop = 0 #??

    while failsafe:

        for epoch in range(num_epochs):

            logger.info('Epoch: {}'.format(epoch))
            logger.info("EXP: {}".format(args.save) )

            phases = []
            if status == 'search':
                phases = ['train', 'val']
            else:
                # while evaluating, add dev set to train also
                phases = ['train', 'val', 'test']

            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == 'train':
                    if not isinstance(scheduler, aux.LRCosineAnnealingScheduler):
                        scheduler.step()
                    if architect is not None:
                        architect.log_learning_rate(logger)

                    model.train()  # Set model to training mode
                    list_preds = []
                    list_target = []

                elif phase == 'val':
                    if status == 'eval':
                        if not isinstance(scheduler, aux.LRCosineAnnealingScheduler):
                            scheduler.step()
                    model.train()
                    list_preds = []
                    list_target = []
                else:
                    model.eval()  # Set model to evaluate mode
                    list_preds = []
                    list_target = []

                running_loss = 0.0
                running_mse = init_mse

                with tqdm(dataloaders[phase]) as t:
                    # Iterate over data.
                    for data in dataloaders[phase]:#############################


                       # image, target = image.to(device), target.to(device, non_blocking=True)
                      #  x_search, target_search = next(iter(valid_loader))  # [b, 3, 32, 32], [b] #######??????????????????????????????????????????????????/
                      #  x_search, target_search = x_search.to(device), target_search.to(device, non_blocking=True)

                        # get the inputs
                        image, digit, target = data #####['image'], data['digit'], data['target']
                        print("shape of im digit target:",image.shape, digit.shape, target.shape)
                        # device
                        image = image.to(device)
                        digit = digit.to(device)
                        target = target.to(device)

                        print("shape of im digit target:",image.shape, digit.shape, target.shape)

                        if status == 'search' and (phase == 'val' or phase == 'test'):
                            architect.step((digit, image), target, logger)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train' or (phase == 'val' and status == 'eval')):
                            output = model((digit, image))

                            if isinstance(output, tuple):
                                output = output[-1]

                            _, output = torch.max(output, 1)  ############preds>output


                            """ target = target.unsqueeze(1), before passing target to criterion,  changed the target tensor size from [16] to [16,1]. Doing it solved the issue.
                            Furthermore, I also needed to do target = target.float() before passing it to criterion,
                            because our outputs are in float. Besides, there was another error in the code.
                            I was using sigmoid activation function in the last layer, but I shouldn’t because the criterion I am using already comes with sigmoid builtin.         """
                            #이렇게 해도 되는건지
                            target = target.expand(-1,2) #torch

                            target=target.reshape(16,-1)
                            target=target.squeeze(1)
                            print(target.shape)

                            print(output.shape,target.shape) #torch.Size([16]) torch.Size([8, 1])

                            loss = criterion(output, target)# Target size (torch.Size([8, 1])) must be the same as input size (torch.Size([16]))

                            preds_th = torch.sigmoid(output) < th_mse2

                            # backward + optimize only if in training phase
                            if phase == 'train' or (phase == 'val' and status == 'eval'):
                                if isinstance(scheduler, aux.LRCosineAnnealingScheduler):
                                    scheduler.step()
                                    scheduler.update_optimizer(optimizer)

                                loss = Variable(loss, requires_grad = True)#######추가element 0 of tensors does not require grad and does not have a grad_fn


                                loss.backward()
                                optimizer.step()

                            # if phase == 'val':
                            list_preds.append(preds_th.cpu())
                            list_target.append(target.cpu())

                        # statistics
                        running_loss += loss.item() * image.size(0)

                        batch_pred_th = preds_th.data.cpu().numpy()
                        batch_true = target.data.cpu().numpy()

                        batch_mse = mean_squared_error(batch_pred_th, batch_true)#, zero_division=1)

                        postfix_str = 'batch_loss: {:.03f}, batch_mse: {:.03f}'.format(loss.item(), batch_mse)
                        t.set_postfix_str(postfix_str)
                        t.update()

                epoch_loss = running_loss / dataset_sizes[phase]

                y_pred = torch.cat(list_preds, dim=0).numpy()
                y_true = torch.cat(list_target, dim=0).numpy()

                # epoch_mse = mean_squared_error(y_true, y_pred, average='macro', zero_division=1)
                epoch_mse = mean_squared_error(y_true, y_pred)#, zero_division=1)

                logger.info('{} Loss: {:.4f}, {} mse: {:.4f}'.format(
                    phase, epoch_loss, epoch_mse))

                if parallel:
                    num_params = 0

                    for reshape_layer in model.module.reshape_layers:

                        num_params += count_parameters(reshape_layer) ###

                    num_params += count_parameters(model.module.fusion_net)
                    logger.info("Fusion Model Params: {}".format(num_params) )

                    genotype = model.module.genotype()

                else:
                    num_params = 0
                    for reshape_layer in model.reshape_layers:
                        num_params += count_parameters(reshape_layer)

                    num_params += count_parameters(model.fusion_net)
                    logger.info("Fusion Model Params: {}".format(num_params) )

                    genotype = model.genotype()
                logger.info(str(genotype))

                if phase == 'train' and epoch_loss != epoch_loss:
                    logger.info("Nan loss during training, escaping")
                    model.eval()
                    return least_mse

                if phase == 'val' and status == 'search':
                    if epoch_mse < least_mse:
                        least_mse = epoch_mse

                        best_genotype = copy.deepcopy(genotype)
                        best_epoch = epoch
                        # best_model_sd = copy.deepcopy(model.state_dict())

                        if parallel:
                            save(model.module, os.path.join(args.save, 'best', 'best_model.pt'))

                        else:
                            save(model, os.path.join(args.save, 'best', 'best_model.pt'))

                        best_genotype_path = os.path.join(args.save, 'best', 'best_genotype.pkl')
                        save_pickle(best_genotype, best_genotype_path) # best geotype 변수를 path에 저장해라

                if phase == 'test':
                    if epoch_mse < least_test_mse:
                        least_test_mse = epoch_mse
                        best_test_genotype = copy.deepcopy(genotype)
                        best_test_epoch = epoch

                        if parallel:
                            save(model.module, os.path.join(args.save, 'best', 'best_test_model.pt'))
                        else:
                            save(model, os.path.join(args.save, 'best', 'best_test_model.pt'))

                        best_test_genotype_path = os.path.join(args.save, 'best', 'best_test_genotype.pkl')
                        save_pickle(best_test_genotype, best_test_genotype_path)

            file_name = "epoch_{}".format(epoch)
            file_name = os.path.join(args.save, "architectures", file_name)
            plotter.plot(genotype, file_name, task='stock')

            logger.info("Current best dev {} mse: {}, at training epoch: {}".format(least_mse, best_epoch) )
            logger.info("Current best test {} mse: {}, at training epoch: {}".format(least_test_mse, best_test_epoch) )

        if least_mse != least_mse and num_epochs == 1 and cont_overloop < 1:
            failsafe = True
            logger.info('Recording a NaN mse, training for one more epoch.')
        else:
            failsafe = False

        cont_overloop += 1

    if least_mse != least_mse:
        least_mse = 0.0

    if status == 'search':
        return least_mse, best_genotype

    else:
        return least_test_mse, best_test_genotype

def test_track_mse(  model, criterion, dataloaders,
                           dataset_sizes, device,
                           parallel, logger, args,
                           mse_type = 'weighted', init_mse=1, th_mse2=0.0001):

    best_test_genotype = None
    least_test_mse = init_mse
    best_test_epoch = 0

    model.eval()  # Set model to evaluate mode
    list_preds = []
    list_target = []

    running_loss = 0.0
    running_mse = init_mse
    phase = 'test'

    with tqdm(dataloaders[phase]) as t:
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            image, digit, target = data['image'], data['digit'], data['target']
            # device
            image = image.to(device)
            digit = digit.to(device)
            target = target.to(device)

            output = model((digit, image)) ##
            if isinstance(output, tuple):
                output = output[-1]

            _, output = torch.max(output, 1) ####pres>output
            loss = criterion(output, target)
            preds_th = torch.sigmoid(output) < th_mse2
            # if phase == 'val':
            list_preds.append(preds_th.cpu())
            list_target.append(target.cpu())

            # statistics
            running_loss += loss.item() * image.size(0)

            batch_pred_th = preds_th.data.cpu().numpy()
            batch_true = target.data.cpu().numpy()
            batch_mse = mean_squared_error(batch_pred_th, batch_true, average='samples')

            postfix_str = 'batch_loss: {:.03f}, batch_mse: {:.03f}'.format(loss.item(), batch_mse)
            t.set_postfix_str(postfix_str)
            t.update()

    epoch_loss = running_loss / dataset_sizes[phase]

    # if phase == 'val':
    y_pred = torch.cat(list_preds, dim=0).numpy()
    y_true = torch.cat(list_target, dim=0).numpy()

    epoch_mse = mean_squared_error(y_true, y_pred) #, average=mse_type) ###??????

    logger.info('{} Loss: {:.4f}, {} mse: {:.4f}'.format(phase, epoch_loss, epoch_mse)) #  mse_type='weighted', ????

    if parallel:
        num_params = 0
        for reshape_layer in model.module.reshape_layers:
            num_params += count_parameters(reshape_layer)

        num_params += count_parameters(model.module.fusion_net)
        logger.info("Fusion Model Params: {}".format(num_params) )
        genotype = model.module.genotype()

    else:
        num_params = 0
        for reshape_layer in model.reshape_layers:
            num_params += count_parameters(reshape_layer)

        num_params += count_parameters(model.fusion_net)
        logger.info("Fusion Model Params: {}".format(num_params) )
        genotype = model.genotype()
    logger.info(str(genotype))
    least_test_mse = epoch_mse
    return least_test_mse