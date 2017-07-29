from data_reader.text_reader import to_np_array


def main():
    '''
    Main assumes that the data file is stored in the subfolder called data.
    :return:
    '''
    file_path = './data/creditcard.csv'
    credit_card_data = to_np_array(file_path)
    print('Loaded the credit card file, it contains {n} rows'.format(n=len(credit_card_data)))

if __name__=='__main__':
    main()