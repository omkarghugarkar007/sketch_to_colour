import tensorflow as tf
import os
import time
import datetime
import matplotlib.pyplot as plt
from Gan_Model import generator, discriminator , train_dataset, test_dataset

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

LAMBDA = 100
epochs = 100

def gen_loss(disc_generated_output,gen_output,target):

    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA*l1_loss)

    return total_gen_loss,gan_loss,l1_loss

def  discriminator_loss(disc_real_output,disc_generated_output):

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    toal_disc_loss = real_loss + generated_loss

    return toal_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dirs = 'Sketch_2_Colour_training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dirs,"ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer, discriminator_optimizer = discriminator_optimizer, generator = generator, discriminator = discriminator)

def generated_img(model,test_input,tar):

    prediction = model(test_input,training = True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar, prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):

        plt.subplot(1,3,i+1)
        plt.title(title[i])
        plt.imshow(display_list[i]*0.5 + 0.5)
        plt.axis('off')

    plt.show()

log_dir = "Sketch2Colour_logs"

summary_writer = tf.summary.create_file_writer(log_dir + 'fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function

def train_step(input_img, target, epoch):

    with tf.GradientTape() as gen_tape, tf.GradientTape as disc_tape:

        gen_output = generator(input_img, training = True)

        disc_real_output = discriminator([input_img,target], training = True)
        disc_generated_output = discriminator([input_img,gen_output], training = True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = gen_loss(disc_generated_output,gen_output,target)

        disc_loss = discriminator_loss(disc_real_output,disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))
    discriminator_optimizer.apply_gradients((zip(discriminator_gradients,discriminator.trainable_variables)))

    with summary_writer.as_default():

        tf.summary.scalar('gen_total_loss',gen_total_loss, epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss,step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_ds, epochs, test_ds):

    for epoch in range(epochs):

        start = time.time()

        for example_input, example_target in test_ds.take(1):
            generated_img(generator,example_input, example_target)

        for n, (input_img,target) in train_ds.enumerate():
            print(".", end='')
            if (n+1)%100 == 0:
                print()
            train_step(input_img, target,epoch)
        print()

        if (epoch + 1)%5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print("Time taken for epoch {} is {} sec \n".format(epoch+1, time.time()-start))

    checkpoint.save(file_prefix=checkpoint_prefix)


fit(train_dataset,epochs,test_dataset)

generator.save_weights("generator.h5")


