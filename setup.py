import os
from setuptools import setup


# modules = ["QtVersions",
#            "constants",
#            "conversions",
#            "do_prtfs",
#            "elements",
#            "elser_particles",
#            "function_probe",
#            "gui",
#            "icosahedral_sphere",
#            "image_manipulation",
#            "parallel",
#            "parse_libconfig",
#            "rotations"
#            "shell_functions"
#            "sphelper"
#            "spimage_tools"
#            "time_tools"
#            "tools"
#            "vtk_tools"]
scripts = ["eke_attenuation_length.py",
           "eke_center_image.py",
           "eke_copy_final.py",
           "eke_copy_good.py",
           "eke_crop_image.py",
           "eke_dic.py",
           "eke_dic_advanced.py",
           "eke_error_reduction.py",
           "eke_ev_to_nm.py",
           "eke_get_errors.py",
           "eke_image_info.py",
           "eke_image_to_png.py",
           "eke_merge_pdf.py",
           "eke_new_to_png.py",
           "eke_nm_to_ev.py",
           "eke_plot_1d.py",
           "eke_plot_2d.py",
           "eke_plot_image.py",
           "eke_plot_log.py",
           "eke_plot_phases.py",
           "eke_plot_image_3d.py",
           "eke_plot_multiple_3d.py",
           "eke_plot_prtf.py",
           "eke_plot_simple.py",
           "eke_pnccd_to_image.py",
           "eke_shift_image.py",
           "eke_show_auto.py",
           "eke_split_pnccd.py",
           "eke_to_png.py",
           "eke_view_pnccd.py",
           "eke_pdb_fix_element.py",
           "eke_hdf5_copy.py",
           "eke_hdf5_del.py",
           "eke_plot_3d.py",
           "eke_plot_radial_average.py",
           "eke_pdb_size.py",
           "eke_remove_all_but_last.py"]

scripts_dir = "Scripts"
scripts_full_path = [os.path.join(scripts_dir, this_script)
                     for this_script in scripts]


setup(name="python-tools",
      version="1.0",
      author="Tomas Ekeberg",
      packages=["eke"],
      package_data={"eke": ["eke/atomsf.dat", "eke/elements.dat"]},
      include_package_data=True,
      scripts=scripts_full_path)
