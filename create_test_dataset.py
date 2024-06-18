import numpy as np 
import PIL.Image
import tensorflow as tf
import io
import glob 

images = []
for filename in glob.glob("/scratch/aj3281/DCR/test_cifar/*"):
    img = PIL.Image.open(filename)
    img = np.array(img)
    images.append(img)


outfile = 'data/true_data_test.npz'
images_np = np.array(images)

with tf.io.gfile.GFile(outfile, "wb") as fout:
    io_buffer = io.BytesIO()

    np.savez_compressed(io_buffer, samples=images_np)
    # else:
    #     np.savez_compressed(io_buffer, samples=images_np, label=class_labels.cpu().numpy())
    fout.write(io_buffer.getvalue())