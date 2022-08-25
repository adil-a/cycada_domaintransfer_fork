import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
# from util.visualizer import Visualizer
from torchvision import utils
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    # visualizer = Visualizer(opt)
    # quit()
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            # visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            # if total_steps % len(dataset) == 0:
            #     visuals = model.get_current_visuals()
            #     imgs = []
            #     for label, image in visuals.items():
            #         imgs.append(image[0])
            #     utils.save_image(torch.stack(imgs), opt.checkpoints_dir + f"/pics/epoch_{epoch}.png")

            #     save_result = total_steps % opt.update_html_freq == 0
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # if total_steps % len(dataset) == 0:
            #     losses = model.get_current_losses()
            #     t = (time.time() - iter_start_time) / opt.batchSize
            #     message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
            #     for k, v in losses.items():
            #         message += '%s: %.3f ' % (k, v)
            #     print(message)
                # visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            # if total_steps % opt.save_latest_freq == 0:
            #     print('saving the latest model (epoch %d, total_steps %d)' %
            #           (epoch, total_steps))
            #     model.save_networks('latest')

            iter_data_time = time.time()
        
        visuals = model.get_current_visuals()
        imgs = []
        for label, image in visuals.items():
            imgs.append(image[0])
        utils.save_image(torch.stack(imgs), opt.checkpoints_dir + f"/pics/epoch_{epoch}.png")

        losses = model.get_current_losses()
        t = (time.time() - epoch_start_time)
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
