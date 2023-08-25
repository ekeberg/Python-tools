import numpy

from eke import diffraction
from eke import elements
from eke import tools


class TemplateMatcher:
    """Compare a 2D diffraction pattern to sphere diffraction to
     find the best fit. It finds the sphere diffraction pattern
     with the maximum scalar product between the radial average
     of the pattern and the template."""
    def __init__(self, npixels: int, wavelength: float,
                 detector_distance: float, pixel_size: float,
                 size_start: float, size_end: float,
                 ntemplates: int):
        self.npixels = npixels
        self.wavelength = wavelength
        self.detector_distance = detector_distance
        self.pixel_size = pixel_size
        self.templates, self.sizes = self._generate_templates(
            size_start, size_end, ntemplates)

    def _template_radial_average(self, size: float):
        """Generate radial average of the diffraction from a sphere
        of the given size"""
        material = elements.MATERIALS["sucrose"]
        sphere_diffraction = abs(diffraction.sphere_diffraction(
            size, material, self.wavelength, (1, 2*self.npixels),
            self.detector_distance, self.pixel_size, 1))**2
        return sphere_diffraction.flatten()[self.npixels:]

    def _generate_templates(self, start_size: float,
                            end_size: float, ntemplates: int):
        """Generate a set of templates from the given size range."""
        templates = numpy.zeros((ntemplates, self.npixels))
        sizes = numpy.zeros(ntemplates)
        size_iterator = numpy.linspace(start_size, end_size, ntemplates)
        for index, size in enumerate(size_iterator):
            templates[index, ...] = self._template_radial_average(size)
            sizes[index] = size
        # Normalize templates
        # templates /= numpy.sqrt((templates**2).sum(axis=1))[:, numpy.newaxis]

        return templates, sizes

    def match(self, pattern, mask=None):
        """Find the template that best matches the pattern."""
        radial_average = tools.radial_average(pattern, mask)[:self.npixels]
        mask_radial_average = tools.radial_average(numpy.float_(mask))
        mask_radial_average = mask_radial_average[:self.npixels] > 0
        radial_average[~mask_radial_average] = 0
        dotprod = (self.templates @ radial_average)
        norm = numpy.sqrt(
            ((self.templates*mask_radial_average)**2).sum(axis=1))
        score = dotprod / norm
        # from IPython import embed; embed()
        best_index = score.argmax()
        self.last_scaling = score[best_index]
        return self.sizes[best_index]
