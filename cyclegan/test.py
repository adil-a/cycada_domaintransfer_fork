import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from torchvision import utils


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # test
    for i, data in enumerate(dataset):
        # if i >= opt.how_many:
        #     break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        for label, im_data in visuals.items():
            image_name = '%05d_%s.png' % (i, label)
            save_path = os.path.join(opt.results_dir, image_name)
            utils.save_image(im_data, save_path)

    # webpage.save()
