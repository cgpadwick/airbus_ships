import argparse
import logging
import helpers
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau, CSVLogger
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
                 gamma=None,
                 alpha=None,
                 wandb_logging=None):

    if wandb_logging:
        wandb.init()
        wandb.config.update(locals())

    if not prefix:
        prefix = str(uuid.uuid1())

    train_img_dir = os.path.join(input_dir, 'train')

    res = helpers.generate_training_and_validation_data(input_dir)

    train_isship_list = res['train_isship_list']
    train_nanship_list = res['train_nanship_list']
    val_isship_list = res['val_isship_list']
    val_nanship_list = res['val_nanship_list']
    train = res['train']
    val = res['val']
    train_df = res['train_df']

    logging.info(train_df.columns)

    use_dropout = False
    if use_dropout_choice == 'true':
        logging.info('using dropout in the model definition')
        use_dropout = True

    model = None
    if model_choice == 'unet-hypercol':
        model = seg_models.unet_with_hypercolumn(use_dropout=use_dropout)
    else:
        raise Exception('unsupported model type')

    cap_num = min(len(train_isship_list), len(train_nanship_list))
    datagen = helpers.data_generator(train_isship_list, train_img_dir=train_img_dir,
                                     train_df=train_df, batch_size=batch_size,
                                     cap_num=cap_num)
    logging.info('loading validation images')
    valgen = helpers.data_generator(val_isship_list, batch_size=50, cap_num=cap_num,
                                    train_img_dir=train_img_dir, train_df=train_df)
    val_x, val_y = next(valgen)

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
    else:
        raise Exception('unsupported loss type')

    log_dir = os.path.join('./', prefix)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=5,
                                       verbose=1, mode='auto', cooldown=2,
                                       min_delta=min_delta, min_lr=1e-7)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_graph=True, write_images=True)

    csv_logger = CSVLogger(os.path.join(log_dir, 'log.csv'))
    callback_list = [reduceLROnPlat, tb_callback, csv_logger]
    if wandb_logging:
        callback_list.append(WandbCallback(monitor='val_loss'))


    model.compile(optimizer=Adam(lr, decay=0.0), loss=loss, metrics=metrics)
    history = model.fit_generator(datagen, steps_per_epoch=num_steps, epochs=epochs,
                                  callbacks=callback_list,
                                  validation_data=(val_x, val_y), workers=1)
    model_filename = os.path.join('./', prefix, 'segmentation_model.h5')
    model.save(model_filename)

    pred_dir = os.path.join('./', prefix, 'val')
    subset_list = val_isship_list[0:num_val_images]
    summary = helpers.output_val_predictions(pred_dir,
                                             subset_list,
                                             model,
                                             train_df,
                                             train_img_dir,
                                             wandb_logging)

    return summary


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=('unet-hypercol', 'unet'), required=False,
                        default='unet-hypercol', help="type of model to use for training")
    parser.add_argument("--loss", choices=('dice', 'focalloss', 'iou', 'custom'), required=False,
                        default='dice', help="type of loss function to use for training")
    parser.add_argument("--epochs", required=False, default=300, type=int,
                        help="number of epochs to train for")
    parser.add_argument("--lr", required=False, default=0.001, type=float,
                        help="initial learning rate")
    parser.add_argument("--batch-size", dest='batch_size', required=False, default=16,
                        help="batch size")
    parser.add_argument("--num-steps", dest='num_steps', required=False, default=250, type=int,
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
    parser.add_argument("--gamma", dest='gamma', required=False, default=2.0, type=float,
                        help="the value of gamma to use for focal loss, only applies to focal loss")
    parser.add_argument("--alpha", dest='alpha', required=False, default=0.25, type=float,
                        help="the value of alpha to use for focal loss, only applies to focal loss")
    parser.add_argument("--wandb-logging", dest='wandb_logging', required=False, default='true',
                        type=str, choices=('true', 'false'),
                        help="option to turn on and off wandb logging")
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
                 gamma=args.gamma,
                 alpha=args.alpha,
                 wandb_logging=True if args.wandb_logging == 'true' else False)



