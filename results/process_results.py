# std
import os
import sys
import re
import csv
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_path(filename, dirname=None):
    root = os.path.dirname(os.path.dirname(__file__))
    logger.debug(f'root: {root}')

    path = os.path.join(root, dirname, filename) if dirname else os.path.join(root, filename)
    logger.debug(f'path: {path}')

    return path


def parse_results(path):
    with open(path, 'r') as resultsfile:
        pattern = re.compile(r'(epoch: [0-9]+), (cost_[a-z]+: [0-9]+\.?[0-9]*)')
        results = defaultdict(list)
        for line in resultsfile:
            line = line.strip()
            logger.debug(f'line: {line}')
            record = pattern.findall(line)
            logger.debug(f'record: {record}')
            if record:
                record, = record
                value = float(record[1].split(':')[1].strip())
                logger.debug(f'value: {value}')
                results[record[0]].append(value)
                logger.debug(f'result: {results}')

    logger.debug(f'results: {results}')

    return results


def write_results(path, results):
    with open('rntn_train_validate_and_test_wordnet_baseline.csv', mode='w') as resultsfile:
        csv_writer = csv.writer(resultsfile)
        csv_writer.writerow(['cost_training', 'cost_validation', 'cost_test'])
        for epoch in results:
            csv_writer.writerow(results[epoch])


def main():
    path = get_path('rntn_train_validate_and_test_wordnet_baseline.log', 'results')
    logger.info('Parsing results...')
    results = parse_results(path)
    logger.info('Parsing results complete!')
    logger.info('Writing results...')
    write_results(path, results)
    logger.info('Writing results complete!')


if __name__ == '__main__':
    logger.info('START!')
    main()
    logger.info('DONE!')
