"""Calls the repeat reconstruction script to do many reconstructions. This is obsolete for use
in Uppsala since multiple reconstructions are preferably run on the GPU cluster."""
import os
import re

class Reconstructer(object):
    """Calls the repeat reconstruction script to do many reconstructions. Made to be used
    together with the MultipleReconstructions class. Takes the configuration file to use,
    the intensities file, a directory in which to put the result, the number of reconstructions
    to perform and a dictionary containing changes to make to the conf file (as "name" : "value")."""
    def __init__(self, conf_file, ints_name, base_dir, n_recs, conf_dict):
        self._ints_name = ints_name
        self._base_dir = base_dir
        self._n_recs = n_recs
        self._conf_dict = conf_dict
        self._rec_name = os.path.basename(self._ints_name)[:-3]
        self._rec_dir = '%s/%s' % (self._base_dir, self._rec_name)
        if not os.path.isdir(self._rec_dir):
            os.mkdir(self._rec_dir)
        os.chdir(self._rec_dir)
        self._conf_file = self._edit_conf(conf_file)

    def _edit_conf(self, conf_file):
        """Update the configurations file based on the _conf_dict variables."""
        with open(conf_file, "r") as file_handle:
            lines = file_handle.readlines()
        index = [i for i in range(len(lines)) if re.search('intensities_file', lines[i])][0]
        #index = lines.index('  intensities_file =;\n')
        lines[index] = '  intensities_file = \"%s\";\n' % self._ints_name
        for key in self._conf_dict.keys():
            index = [i for i in range(len(lines)) if re.search('%s =' % key, lines[i])][0]
            re_result = re.search('^(.*)=', lines[index])
            if re_result:
                key_name = re_result.groups()[0] #this code looks unnecesarily complicated
                lines[index] = '%s = %s;\n' % (key_name, self._conf_dict[key])
        new_file = '%s/uwrapc.conf' % (self._rec_dir)
        with open(new_file, "w") as file_handle:
            file_handle.writelines(lines)
        return new_file

    def start(self):
        """Starts the reconstruction"""
        os.system('/usr/local/bin/repeat_reconstruction.pl %d' % self._n_recs)

class MultipleReconstructions(object):
    """Performs multiple reconstructions of multiple files using the Reconstructer class."""
    def __init__(self, conf_file, ints_dir, base_dir, n_recs, conf_dict):
        self._conf_file = conf_file
        self._ints_dir = ints_dir
        self._base_dir = base_dir
        self._n_recs = n_recs
        self._conf_dict = conf_dict
        self._files = self.get_file_list()

    def get_file_list(self):
        """Create a list of paths to all h5 files."""
        files = os.listdir(self._ints_dir)
        files = ['%s/%s' % (self._ints_dir, f) for f in files if re.search('\.h5$', f)]
        files.sort()
        return files

    def start(self):
        """Run all reconstructions."""
        for file_name in self._files:
            print 'Reconstruct %s' % file_name
            Reconstructer(self._conf_file, file_name, self._base_dir, self._n_recs, self._conf_dict).start()

if __name__ == "__main__":
    TEST_CONF_DICT = {'max_iterations' : '5000',
                      'work_directory' : '\".\"',
                      'output_period' : '1000',
                      'log_output_period' : '500'}

    MultipleReconstructions('/data/LCLS2011/r0138/all/uwrapc.conf',
                            '/home/ekeberg/Work/python/mimi_finder/preprocess/processed_for_reconstruction',
                            '/data/LCLS2011/r0138/all',
                            30, TEST_CONF_DICT).start()
