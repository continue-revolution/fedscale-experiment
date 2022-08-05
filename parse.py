import pathlib
from absl import app, flags
# flags.DEFINE_string('filename', 'out', 'filename of output')

def main(_):
    lines = []
    for i in [1, 2, 4, 8, 16, 32]:
        contents = pathlib.Path(f'out{i}').read_text().split('\n')[1:-1]
        contents = map(lambda x: '\t'.join([x.split('\t')[0] + f'-{i}', x.split('\t')[1]]), contents)
        lines.extend(contents)
    # filename = flags.FLAGS.filename
    # file = pathlib.Path(filename)
    # contents = file.read_text()
    # lines = contents.split('\n')
    datas = {}
    # first_line = True
    for line in lines:
        # if first_line:
        #     first_line = False
        #     continue
        if line:
            category, data = line.split('\t')
            key = category.split('-')[0]
            if key not in datas:
                datas[key] = [(category, float(data))]
            else:
                datas[key].append((category, float(data)))
    for _list in datas.values():
        _list.sort(key=lambda x: x[1])
        _list = map(lambda x: f'{x[0]}\t{x[1]}', _list)
        print('\n'.join(_list))
        print('---')


if __name__ == '__main__':
    app.run(main)
