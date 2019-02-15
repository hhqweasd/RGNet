import time
from data.data_loader import get_data_loader
from models.models import create_model
from option_parser import TrainingOptionParser
from utils.visualizer import Visualizer

parser = TrainingOptionParser()
opt = parser.parse_args()

data_loader = get_data_loader(opt)

print("[INFO] batch size : {}".format(opt.batch_size))
print("[INFO] training batches : {}".format(len(data_loader)))

model = create_model(opt)
epoch_count = 0
#model.load(epoch_count)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch):
#for epoch in range(epoch_count, opt.epoch):
    # if epoch < 300:
    #     continue
    epoch_start_time = time.time()
    iter_count = 0

    for i, data in enumerate(data_loader):
        batch_start_time = time.time()
        total_steps += opt.batch_size
        iter_count += opt.batch_size
        # data : list

        # GibbsNet
        # model.set_input(data[0])
        # model.optimize_parameters()
        # batch_end_time = time.time()
        #
        # if iter_count % opt.print_freq == 0:
        #     errors = model.get_losses()
        #     visualizer.print_current_errors(epoch, iter_count, errors, (batch_end_time - batch_start_time))
        #
        # if total_steps % opt.plot_freq == 0:
        #     save_result = total_steps % opt.plot_freq == 0
        #     visualizer.display_current_results(model.get_visuals(), int(total_steps/opt.plot_freq), save_result)
        #     if opt.display_id > 0:
        #         visualizer.plot_current_errors(epoch, total_steps, errors)

        # RGNet part
        model.set_input(data[0])
        model.optimize_parameters(opt.inferring_count)
        batch_end_time = time.time()

        if iter_count % opt.print_freq == 0:
            errors = model.get_losses()
            print('epoch=',epoch,', iter_count=', iter_count,'errors=', errors, 'time=', (batch_end_time - batch_start_time))
        	#visualizer.print_current_errors(epoch, iter_count, errors, (batch_end_time - batch_start_time))

        if total_steps % opt.plot_freq == 0:
            save_result = total_steps % opt.plot_freq == 0
            visualizer.display_current_results(model.get_visuals(), int(total_steps/opt.plot_freq), save_result)
            #if opt.display_id > 0:
            #    visualizer.plot_current_errors(epoch, total_steps, errors)

    # model.remove(epoch_count)
    epoch_count += 1
    if epoch_count % 5 == 0:
        model.save(epoch_count)
