from data_generator import get_iterators

def show_img(img_iterator):
    import matplotlib.pyplot as plt
    # img_batch_data.reset()
    data_batch = img_iterator.next()
    data = data_batch.data[0]
    plt.figure()
    for i in range(8):
        _image = data[i].astype('uint8').asnumpy().transpose((1, 2, 0))
        plt.subplot(2, 4, i + 1)
        plt.imshow(_image)
    plt.show()

if __name__ == "__main__":
    (train,val) = get_iterators()
    show_img(val)