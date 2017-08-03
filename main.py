from data_reader.text_reader import read_to_numpy_array
from sampler import split_into_training_and_validation
import logging

def main():
    '''
    Main assumes that the data file is stored in the sub-folder called data.
    '''

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    # Load from relative folder
    logging.info('Loading data...')
    file_path = './data/creditcard.csv'
    credit_card_data = read_to_numpy_array(file_path)
    print('Loaded the credit card file, it contains {n} rows'.format(n=len(credit_card_data)))

    # Split
    print('')
    (training_set, validation_set) = split_into_training_and_validation(credit_card_data, 0.8)
    print('Training set has {n} rows.'.format(n=len(training_set)))
    print('Validation set has {n} rows.'.format(n=len(validation_set)))



if __name__=='__main__':
    main()