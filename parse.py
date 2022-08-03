from distutils.command.build_scripts import first_line_re
import pathlib
from absl import app, flags
flags.DEFINE_string('filename', 'out', 'filename of output')

def main(_):
    filename = flags.FLAGS.filename
    file = pathlib.Path(filename)
    contents = file.read_text()
    lines = contents.split('\n')
    datas = {}
    first_line = True
    for line in lines:
        if first_line:
            first_line = False
            continue
        if line:
            category, data, _ = line.split('\t')
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
