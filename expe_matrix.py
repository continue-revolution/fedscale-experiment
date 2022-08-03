from absl import app, flags
import time
import random
import torch
flags.DEFINE_integer('iter', 100, 'number of iterations')
flags.DEFINE_integer('start', 0, 'start of plot')


def main(_):
    num_iteration = flags.FLAGS.iter

    with torch.no_grad():
        for thread_num in [1, 2, 4, 16, 32]:
            torch.set_num_threads(thread_num)
            for device in ['cpu', 'cuda']:
                random.seed(100)
                torch.manual_seed(100)
                print(f'{device}-{thread_num}', end="")
                model = torch.rand(size=(1000, 3, 100, 100), device=device)
                _model = torch.rand(size=(1000, 3, 100, 100), device=device)
                time_list = []
                for _ in range(num_iteration):
                    _model += random.random() * model
                    time_list.append(time.time())
                print(f"\t{time_list[-1] - time_list[0]}")


if __name__ == '__main__':
    app.run(main)
