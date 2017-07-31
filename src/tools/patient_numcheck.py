import os


class PatientNumChecker:
    def __init__(self, data_directory):
        self.path = os.path.join(data_directory, 'raw')

    def check(self, pat_no):
        """
        Checks in all 3 folders if the patient number is already in use.
        """
        if len(pat_no) != 4 or not pat_no.isdigit():
            print(f'{pat_no} is not a valid patient number!')
            return

        if self.check_folder('train', pat_no):
            print(f'{pat_no} found in folder TRAIN')
        elif self.check_folder('test', pat_no):
            print(f'{pat_no} found in folder TEST')
        elif self.check_folder('validation', pat_no):
            print(f'{pat_no} found in folder VALIDATION')
        else:
            print(f'{pat_no} is not yet used')

    def check_folder(self, direc, pat_no):
        """
        The patient number to check for.
        :param direc: the directory to look into (can be test/train/validation)
        :param pat_no: the patient number
        :return: true iff the number was found
        """
        direc = os.path.join(self.path, direc)
        patients = os.listdir(direc)
        return any(pat_no in pat for pat in patients)


if __name__ == "__main__":
    pat_in = 'nothing yet'
    pat_check = PatientNumChecker('data/')
    while True:
        pat_in = input('Which patient number should be tested?\n')
        if any(pat_in in exit_term for exit_term in {'q', 'e', 'quit', 'Q'}):
            print('Terminating...')
            break
        pat_check.check(pat_in)
