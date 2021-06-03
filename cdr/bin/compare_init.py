import sys
import argparse

def parse_settings(path):
    settings = {}
    with open(path, 'r') as f:
        in_settings = False
        for line in f:
            if line.strip().startswith('MODEL SETTINGS:'):
                in_settings = True
            elif in_settings:
                if not line.strip():
                    break
                k, v = line.strip().split(': ')
                settings[k] = v

    return settings

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Identify hyperparameters that differ between any two saved CDR models
    ''')
    argparser.add_argument('path1', help='Path to output directory of model 1.')
    argparser.add_argument('path2', help='Path to output directory of model 2.')
    args = argparser.parse_args()

    path1 = args.path1 + '/initialization_summary.txt'
    path2 = args.path2 + '/initialization_summary.txt'

    settings1 = parse_settings(path1)
    settings2 = parse_settings(path2)

    in_1_only = []
    in_2_only = []
    in_both_but_different = []
    for k in settings1:
        setting1 = settings1[k]
        if k in settings2:
            setting2 = settings2.pop(k)
            if setting1 != setting2:
                in_both_but_different.append((k, setting1, setting2))
        else:
            in_1_only.append((k, setting1))

    for k in settings2:
        in_2_only.append((k, settings2[k]))

    if len(in_1_only):
        print('Settings only found in File 1:')
        for k, v in in_1_only:
            print(k, ': ', v)
        print()

    if len(in_2_only):
        print('Settings only found in File 2:')
        for k, v in in_2_only:
            print(k, ': ', v)
        print()

    if len(in_both_but_different):
        print('Settings that differ between File 1 and File 2:')
        for k, v1, v2 in in_both_but_different:
            print(k, ': ', v1, ', ', v2)
        print()

    if not len(in_1_only + in_2_only + in_both_but_different):
        print('Settings identical!')