"""Detection model trainer.

This file provides a generic training method to train a
DetectionModel.
"""
import datetime
import os
import tensorflow as tf
import time

from avod.builders import optimizer_builder
from avod.core import trainer_utils
from avod.core import summary_utils

slim = tf.contrib.slim


def train(model, train_config):
    """Training function for detection models.

    Args:
        model: The detection model object.
        train_config: a train_*pb2 protobuf.
            training i.e. loading RPN weights onto AVOD model.
    """

    model = model
    train_config = train_config
    # Get model configurations
    model_config = model.model_config

    # Create a variable tensor to hold the global step
    global_step_tensor = tf.Variable(
        0, trainable=False, name='global_step')

    #############################
    # Get training configurations
    #############################
    max_iterations = train_config.max_iterations
    summary_interval = train_config.summary_interval
    checkpoint_interval = \
        train_config.checkpoint_interval
    max_checkpoints = train_config.max_checkpoints_to_keep

    paths_config = model_config.paths_config
    logdir = paths_config.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    checkpoint_dir = paths_config.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = checkpoint_dir + '/' + \
        model_config.checkpoint_name

    global_summaries = set([])

    # The model should return a dictionary of predictions
    prediction_dict = model.build()

    summary_histograms = train_config.summary_histograms
    summary_img_images = train_config.summary_img_images
    summary_bev_images = train_config.summary_bev_images

    ##############################
    # Setup loss
    ##############################
    losses_dict, total_loss = model.loss(prediction_dict)


    ##############################################################################################
    # TODO PROJECT: select trainable variables to set gradient to 0(var0)

    var1 = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mix_of_experts')]
    var0 = [var for var in tf.trainable_variables()]
    var_all_but_var1 = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    
    for var in var1:
        var0.remove(var)
        var_all_but_var1.remove(var)

    ##############################################################################################

    # Optimizer
    # training_optimizer = optimizer_builder.build(
        # train_config.optimizer,
        # global_summaries,
        # global_step_tensor)

    ##############################################################################################
    # TODO PROJECT: create optimizer with 0 gradient
    training_optimizer0 = tf.train.GradientDescentOptimizer(0.0)
    training_optimizer1 = optimizer_builder.build(
        train_config.optimizer,
        global_summaries,
        global_step_tensor)
    ##############################################################################################

    # Create the train op
    with tf.variable_scope('train_op'):
        # train_op = slim.learning.create_train_op(
            # total_loss,
            # training_optimizer,
            # clip_gradient_norm=1.0,
            # global_step=global_step_tensor)
    ##############################################################################################
    # TODO PROJECT: create training operations
        train_op1 = slim.learning.create_train_op(
            total_loss,
            training_optimizer1,
            variables_to_train=var1,
            clip_gradient_norm=1.0,
            global_step=global_step_tensor)
        train_op0 = slim.learning.create_train_op(
            total_loss, 
            training_optimizer0,
            variables_to_train=var0,
            clip_gradient_norm=1.0,
            global_step=global_step_tensor)
        train_op = tf.group(train_op1,train_op0)

    ##############################################################################################

    # Save checkpoints regularly.
    saver = tf.train.Saver(max_to_keep=max_checkpoints,
                           pad_step_number=True)

    # Add the result of the train_op to the summary
    tf.summary.scalar("training_loss", train_op1) 
    # Add maximum memory usage summary op
    # This op can only be run on device with gpu
    # so it's skipped on travis
    is_travis = 'TRAVIS' in os.environ
    if not is_travis:
        # tf.summary.scalar('bytes_in_use',
        #                   tf.contrib.memory_stats.BytesInUse())
        tf.summary.scalar('max_bytes',
                          tf.contrib.memory_stats.MaxBytesInUse())

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_merged = summary_utils.summaries_to_keep(
        summaries,
        global_summaries,
        histograms=summary_histograms,
        input_imgs=summary_img_images,
        input_bevs=summary_bev_images
    )

    allow_gpu_mem_growth = train_config.allow_gpu_mem_growth
    if allow_gpu_mem_growth:
        # GPU memory config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = allow_gpu_mem_growth
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()

    # Create unique folder name using datetime for summary writer
    datetime_str = str(datetime.datetime.now())
    logdir = logdir + '/train'
    train_writer = tf.summary.FileWriter(logdir + '/' + datetime_str,
                                         sess.graph)

    # Create init op
    init = tf.global_variables_initializer()

    # Continue from last saved checkpoint
    if not train_config.overwrite_checkpoints:
        trainer_utils.load_checkpoints(checkpoint_dir,
                                       saver)
        if len(saver.last_checkpoints) > 0:
            checkpoint_to_restore = saver.last_checkpoints[-1]
            saver.restore(sess, checkpoint_to_restore)
        else:

            # Initialize the variables
            # sess.run(init)
            ##############################################################################################
            # TODO PROJECT: take checkpoints from original avod model
            checkpoint_path_start = "/storage/remote/atcremers62/avod_moe/avod/data/outputs/pyramid_cars_with_aug_example/checkpoints/start/pyramid_cars_with_aug_example-00100000"
            # variables_to_restore = slim.get_variables_to_restore(include=var0, exclude=var1)
            variables_to_restore = dict()
            for var in var_all_but_var1:
                # candidates = slim.get_variables(var.name)
                # for candidate in candidates:
                    # print(candidate.op.name)
                    # print(var.op.name)
                variables_to_restore[var.op.name] = slim.get_unique_variable(var.op.name)

            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                checkpoint_path_start, variables_to_restore)
            sess.run(init)
            sess.run(init_assign_op, init_feed_dict)
            # raise Exception("init successfully!")
            ##############################################################################################
    else:


        # Initialize the variables
        sess.run(init)

    # Read the global step if restored
    global_step = tf.train.global_step(sess,
                                       global_step_tensor)
    print('Starting from step {} / {}'.format(
        global_step, max_iterations))


    # Main Training Loop
    last_time = time.time()
    for step in range(global_step, max_iterations + 1):

        # Save checkpoint
        if step % checkpoint_interval == 0:
            global_step = tf.train.global_step(sess,
                                               global_step_tensor)

            saver.save(sess,
                       save_path=checkpoint_path,
                       global_step=global_step)

            print('Step {} / {}, Checkpoint saved to {}-{:08d}'.format(
                step, max_iterations,
                checkpoint_path, global_step))

        # Create feed_dict for inferencing
        feed_dict = model.create_feed_dict()


        # Write summaries and train op
        if step % summary_interval == 0:
            current_time = time.time()
            time_elapsed = current_time - last_time
            last_time = current_time

            train_op_loss, summary_out = sess.run(
                [train_op1, summary_merged], feed_dict=feed_dict)
            print(train_op_loss)

            print('Step {}, Total Loss {:0.3f}, Time Elapsed {:0.3f} s'.format(
                step, train_op_loss, time_elapsed))
            train_writer.add_summary(summary_out, step)

        else:
            # Run the train op only
            sess.run(train_op1, feed_dict)

    # Close the summary writers
    train_writer.close()