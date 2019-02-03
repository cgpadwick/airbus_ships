import argparse
import logging
import generate_submission
import helpers
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
import math
import re
import seg_models
import shutil
import os
import uuid
import wandb
from wandb.keras import WandbCallback


def run_training(model_choice=None,
                 loss_choice=None,
                 epochs=None,
                 lr=None,
                 batch_size=None,
                 num_steps=None,
                 prefix=None,
                 num_val_images=None,
                 input_dir=None,
                 min_delta=None,
                 use_dropout_choice=None,
                 lr_schedule=None,
                 lr_decay=None,
                 gamma=None,
                 alpha=None,
                 use_augmentation=None,
                 existing_model=None,
                 wandb_logging=None,
                 wandb_tag=None):

    if not prefix:
        prefix = str(uuid.uuid1())

    if wandb_logging:
        wandb.init(tags=[wandb_tag])
        wandb.config.update(locals())

    train_img_dir = os.path.join(input_dir, 'train')

    segmentation = True if model_choice != 'resnet-classifier' else False

    # If the user passed an existing model, restore it.
    model = None
    if existing_model and os.path.exists(existing_model):
        segmentation = True
        if re.match('classification', os.path.basename(existing_model)):
            segmentation = False
        model = load_model(existing_model, custom_objects={'IoU': helpers.IoU,
                                                           'dice_loss': helpers.dice_loss,
                                                           'focal_loss': helpers.focal_loss,
                                                           'fl': generate_submission.fl,
                                                           'custom_loss': helpers.custom_loss})
        logging.info('Skipping training, restoring model from path {}'.format(existing_model))

    res = helpers.generate_training_and_validation_data(input_dir, segmentation=segmentation)

    train_isship_list = res['train_isship_list']
    train_nanship_list = res['train_nanship_list']
    val_isship_list = res['val_isship_list']
    val_nanship_list = res['val_nanship_list']
    train = res['train']
    val = res['val']
    train_df = res['train_df']

    helpers.balance_lists(train_isship_list, train_nanship_list)

    logging.info('train.shape: {}'.format(train.shape))
    logging.info('len(train_isship_list): {}'.format(len(train_isship_list)))
    logging.info('len(train_nanship_list): {}'.format(len(train_nanship_list)))

    use_dropout = False
    if use_dropout_choice == 'true':
        logging.info('using dropout in the model definition')
        use_dropout = True

    if not num_steps:
        num_steps = int(math.ceil(train.shape[0] / float(batch_size)))
        logging.info('num steps per epoch computed to be: {}'.format(num_steps))

    # If the model hasn't been restored by now, then the user wants us to train a
    # new model.
    if not model:
        if model_choice == 'unet-hypercol':
            model = seg_models.unet_with_hypercolumn(use_dropout=use_dropout)
        elif model_choice == 'unet-resnet':
            model = seg_models.unet_with_resnet_encoder()
        elif model_choice == 'resnet-classifier':
            model = seg_models.resnet_classifier()
        else:
            raise Exception('unsupported model type')

        cap_num = min(len(train_isship_list), len(train_nanship_list))
        datagen = helpers.data_generator(train_isship_list,
                                         train_nanship_list,
                                         train_img_dir=train_img_dir,
                                         train_df=train_df, batch_size=batch_size,
                                         cap_num=cap_num,
                                         segmentation=segmentation)
        data_generator = datagen
        if use_augmentation:
            data_generator = helpers.create_aug_gen(datagen, segmentation=segmentation)
            logging.info('Using augmentation during training')

        logging.info('loading validation images')
        valgen = helpers.data_generator(val_isship_list, val_nanship_list,
                                        batch_size=50, cap_num=cap_num,
                                        train_img_dir=train_img_dir, train_df=train_df,
                                        segmentation=segmentation)
        val_x, val_y = next(valgen)

        if loss_choice != 'binary_crossentropy' and model_choice == 'resnet-classifier':
            raise Exception('invalid choice of model and loss function')
        loss = None
        metrics = ['binary_accuracy']
        if loss_choice == 'dice':
            loss = helpers.dice_loss
            metrics.append(helpers.dice_loss)
        elif loss_choice == 'focalloss':
            loss = helpers.focal_loss(gamma=gamma, alpha=alpha)
            metrics.append(helpers.focal_loss(gamma=gamma, alpha=alpha))
        elif loss_choice == 'iou':
            loss = helpers.iou_measure
            metrics.append(helpers.iou_measure)
        elif loss_choice == 'custom':
            loss = helpers.custom_loss
            metrics.append(helpers.custom_loss)
        elif loss_choice == 'binary_crossentropy':
            loss = 'binary_crossentropy'
        else:
            raise Exception('unsupported loss type')

        log_dir = os.path.join('./', prefix)
        tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=0,
                                  write_graph=True, write_images=True)
        chkpt_callback = ModelCheckpoint(os.path.join(log_dir, 'model_checkpoint.h5'))
        csv_logger = CSVLogger(os.path.join(log_dir, 'log.csv'))
        callback_list = [tb_callback, csv_logger, chkpt_callback]
        if wandb_logging:
            callback_list.append(WandbCallback(monitor='val_loss'))

        decay_rate = 0.0
        if lr_schedule == 'reduceonplateau':
            reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                               patience=5,
                                               verbose=1, mode='auto', cooldown=2,
                                               min_delta=min_delta, min_lr=1e-7)
            callback_list.append(reduceLROnPlat)
        else:
            decay_rate = lr_decay

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        model.compile(optimizer=Adam(lr, decay=decay_rate), loss=loss, metrics=metrics)
        history = model.fit_generator(data_generator,
                                      steps_per_epoch=num_steps,
                                      epochs=epochs,
                                      callbacks=callback_list,
                                      validation_data=(val_x, val_y), workers=1)

        model_name = 'segmentation_model.h5' if segmentation else 'classification_model.h5'
        model_filename = os.path.join('./', prefix, model_name)
        model.save(model_filename)

    pred_dir = os.path.join('./', prefix, 'val')
    subset_list = val_isship_list[0:num_val_images]
    if segmentation:
        summary = helpers.output_val_predictions(val_dir=pred_dir,
                                             val_list=subset_list,
                                             model=model,
                                             train_df=train_df,
                                             train_img_dir=train_img_dir,
                                             wandb_logging=wandb_logging)
    else:
        subset_list = val_isship_list[0:num_val_images // 2] + val_nanship_list[0:num_val_images // 2]
        summary = helpers.output_val_predictions_for_classification(val_dir=pred_dir,
                                             val_list=subset_list,
                                             model=model,
                                             train_df=train_df,
                                             train_img_dir=train_img_dir,
                                             wandb_logging=wandb_logging)
    return summary


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=('unet-hypercol', 'unet-resnet', 'resnet-classifier'),
                        required=False, default='unet-hypercol',
                        help="type of model to use for training")
    parser.add_argument("--loss",
                        choices=('dice', 'focalloss', 'iou', 'custom', 'binary_crossentropy'),
                        required=False,
                        default='dice',
                        help="type of loss function to use for training")
    parser.add_argument("--epochs", required=False, default=300, type=int,
                        help="number of epochs to train for")
    parser.add_argument("--lr", required=False, default=0.001, type=float,
                        help="initial learning rate")
    parser.add_argument("--batch-size", dest='batch_size', required=False, default=16, type=int,
                        help="batch size")
    parser.add_argument("--num-steps", dest='num_steps', required=False, default=None, type=int,
                        help="number of steps per epoch")
    parser.add_argument("--prefix", required=False, default=None,
                        help="prefix for location to store results")
    parser.add_argument("--num-val-images", dest='num_val_images', required=False, default=1000,
                        type=int, help="prefix for location to store results")
    parser.add_argument("--input-dir", dest='input_dir', required=False,
                        default='/raid/cgp/data/airbus_ships/',
                        help="input directory location")
    parser.add_argument("--min-delta", dest='min_delta', required=False,
                        default=0.001, type=float,
                        help="min delta parameter for ReduceLROnPlateau callback")
    parser.add_argument("--use-dropout", dest='use_dropout', required=False,
                        choices=('true', 'false'),
                        default='true',
                        help='use dropout in the model definition (or not)')
    parser.add_argument("--use-augmentation", dest='use_augmentation', required=False,
                        choices=('true', 'false'),
                        default='false',
                        help='use augmentation during training (or not)')
    parser.add_argument("--lr-schedule", dest='lr_schedule', required=False,
                        choices=('reduceonplateau', 'decay'),
                        default='reduceonplateau',
                        help='strategy for learning rate decay')
    parser.add_argument("--lr-decay", dest='lr_decay', required=False,
                        default=0.1, type=float,
                        help="learning rate decay for lr-schedule decay")
    parser.add_argument("--gamma", dest='gamma', required=False, default=2.0, type=float,
                        help="the value of gamma to use for focal loss, only applies to focal loss")
    parser.add_argument("--alpha", dest='alpha', required=False, default=0.25, type=float,
                        help="the value of alpha to use for focal loss, only applies to focal loss")
    parser.add_argument("--existing-model", dest='existing_model', required=False, default=None, type=str,
                        help="pathname to an existing model.  If passed then training will be skipped and evaluation"
                             " will be performed.")
    parser.add_argument("--wandb-logging", dest='wandb_logging', required=False, default='true',
                        type=str, choices=('true', 'false'),
                        help="option to turn on and off wandb logging")
    parser.add_argument("--wandb-tag", dest='wandb_tag', required=False, default=None, type=str,
                        help="tag the run in wandb with a string")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    summary = run_training(model_choice=args.model,
                 loss_choice=args.loss,
                 epochs=args.epochs,
                 lr=args.lr,
                 batch_size=args.batch_size,
                 num_steps=args.num_steps,
                 prefix=args.prefix,
                 num_val_images=args.num_val_images,
                 input_dir=args.input_dir,
                 min_delta=args.min_delta,
                 use_dropout_choice=args.use_dropout,
                 lr_schedule=args.lr_schedule,
                 lr_decay=args.lr_decay,
                 gamma=args.gamma,
                 alpha=args.alpha,
                 use_augmentation=True if args.use_augmentation == 'true' else False,
                 existing_model=args.existing_model,
                 wandb_logging=True if args.wandb_logging == 'true' else False,
                 wandb_tag=args.wandb_tag)



