import numpy as np
import math
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats

class FittingModels:

    def create_dataPoints(self, image1, image2):
        #Creates a list of data points given two images
        #:param image1: first image
        #:param image2: second image
        #:return: a list of data points
        #"""

      datapoints=[]
      image1_intensities=[]
      image2_intensities=[]

      image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
      image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

      for x in range(0,image1.shape[0]):
          for y in range(0,image1.shape[1]):
              i1=image1[x,y]
              image1_intensities.append(i1)

      for a in range(0,image2.shape[0]):
          for b in range(0,image2.shape[1]):
              i2=image2[a,b]
              image2_intensities.append(i2)

      # datapoints consisting of [I1,I2]
      for i, j in zip(range(len(image1_intensities)), range(len(image2_intensities))):
        datapoint = []
        datapoint.append(image1_intensities[i])
        datapoint.append(image2_intensities[j])
        datapoints.append(datapoint)

      return datapoints

    def plot_data(self, data_points):
        """ Plots the data points
        :param data_points:
        :return: an image
        """
        I1_intensities = []
        I2_intensities = []
        for i in range(0, len(data_points)):
            I1_intensities.append(data_points[i][0])
            I2_intensities.append(data_points[i][1])

        plt_image = self.get_image(I1_intensities,I2_intensities,'None')
        return plt_image

    def fit_line_ls(self, data_points, threshold,image2):
        """ Fits a line to the given data points using least squares
        :param data_points: a list of data points
        :param threshold: a threshold value (if > threshold, imples outlier)
        :return: a tuple containing the followings:
                    * An image showing the line along with the data points
                    * The thresholded image
                    * A segmented image
        """
        image1_intensities = []
        image2_intensities= []

        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

        for i in range(0, len(data_points)):
            image1_intensities.append(data_points[i][0])
            image2_intensities.append(data_points[i][1])

        #using formulae to obtain slope and intercept of the line that best fits the model
        I1_mean = np.mean(image1_intensities)
        I2_mean = np.mean(image2_intensities)

        count= len(image1_intensities)
        nm=0
        dm=0
        for i in range(count):
            nm += (image1_intensities[i] - I1_mean) * (image2_intensities[i] - I2_mean)
            dm +=(image1_intensities[i]-I1_mean)**2

        slope = nm / dm
        intercept = I2_mean - (slope * I1_mean)
        print("The parameters (slope and intercept) obtained for least squares are {},{}".format(slope, intercept))

        max_x = np.max(image1_intensities) + 100
        min_x = np.min(image1_intensities) - 100
        x = np.linspace(min_x, max_x,1000)
        y = intercept + slope * x

        # Ploting Regression line and scatter points
        fig = plt.figure()
        plot = fig.add_subplot(111)
        plot.plot(x, y, color='#58b970', label='Regression Line')
        plot.scatter(image1_intensities, image2_intensities, c='#ef5423', label='Scatter Plot')
        plt.xlabel('I1 intensities')
        plt.ylabel('I2 intensities')
        plt.legend()
        fig.canvas.draw()
        line_fitting_ls = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        line_fitting_ls = line_fitting_ls.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        outliers=[]
        image_difference = []

        for i in range(len(image1_intensities)):
            expected_y = intercept + slope * image1_intensities[i]
            dist = abs(image2_intensities[i] - expected_y)
            if(dist>threshold):
                outliers.append(data_points[i])
                image_difference.append(255)
            else:
                image_difference.append(0)

        outliers_x=[]
        outliers_y=[]

        for i in range(len(outliers)):
            outliers_x.append(outliers[i][0])
            outliers_y.append(outliers[i][1])

        thresholded_ls=self.get_image(outliers_x,outliers_y,'None')

        ls_seg_image = np.reshape(image_difference, (image2.shape[0], image2.shape[1]))
        kernel = np.ones((2, 2), np.uint8)
        ls_seg_image = cv2.erode((ls_seg_image * 1.0).astype(np.float32), kernel)
        print("LS")
        return (line_fitting_ls, thresholded_ls, ls_seg_image)

    def fit_line_robust(self, data_points, threshold,image2):
        """ Fits a line to the given data points using robust estimators
        :param data_points: a list of data points
        :param threshold: a threshold value (if > threshold, imples outlier)
        :return: a tuple containing the followings:
                    * An image showing the line along with the data points
                    * The thresholded image
                    * A segmented image
        """
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        data_points_arr=np.array(data_points)
        #using cv2.fitline function to get the parameters of vx,vy,x,y
        [vx, vy, x, y] = cv2.fitLine(data_points_arr, cv2.DIST_L2, 0, 0.01, 0.01)
        slope=vy/vx
        intercept=y-(slope*x)
        print("The parameters (slope and intercept) obtained for robust estimators {},{}".format(slope,intercept))

        image1_intensities = []
        image2_intensities= []
        for i in range(0, len(data_points)):
            image1_intensities.append(data_points[i][0])
            image2_intensities.append(data_points[i][1])

        max_x = np.max(image1_intensities) + 100
        min_x = np.min(image1_intensities) - 100
        x = np.linspace(min_x, max_x, 1000)
        y = intercept + slope * x

        # Ploting regression line and scatter points
        fig = plt.figure()
        plot = fig.add_subplot(111)
        plot.plot(x, y, color='#58b970', label='Regression Line')
        plot.scatter(image1_intensities, image2_intensities, c='#ef5423', label='Scatter Plot')
        plt.xlabel('I1 intensities')
        plt.ylabel('I2 intensities')
        plt.legend()
        fig.canvas.draw()
        line_fitting_robust = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        line_fitting_robust = line_fitting_robust.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        outliers = []
        image_difference=[]

        for i in range(len(image1_intensities)):
            expected_y = intercept + slope * image1_intensities[i]
            dist = abs(image2_intensities[i] - expected_y)
            if (dist > threshold):
                outliers.append(data_points[i])
                image_difference.append(255)
            else:
                image_difference.append(0)

        outliers_x = []
        outliers_y = []

        for i in range(len(outliers)):
            outliers_x.append(outliers[i][0])
            outliers_y.append(outliers[i][1])

        thresholded_robust = self.get_image(outliers_x,outliers_y,'None')

        segmented_robust = np.reshape(image_difference, (image2.shape[0], image2.shape[1]))
        kernel = np.ones((2, 2), np.uint8)
        segmented_robust = cv2.erode((segmented_robust * 1.0).astype(np.float32), kernel)
        print("RO")
        return (line_fitting_robust, thresholded_robust, segmented_robust)

    def fit_gaussian(self, data_points, threshold,image2):
        """ Fits the data points to a gaussian
        :param data_points: a list of data points
        :param threshold: a threshold value (if < threshold, imples outlier)
        :return: a tuple containing the followings:
                    * An image showing the gaussian along with the data points
                    * The thresholded image
                    * A segmented image
        """
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

        image1_intensities = []
        image2_intensities = []
        for i in range(0, len(data_points)):
            image1_intensities.append(data_points[i][0])
            image2_intensities.append(data_points[i][1])

        #finding mean of I1 and I2 intensities
        x_mean=np.mean(image1_intensities)
        y_mean=np.mean(image2_intensities)

        #finding the co variance matrix  and probability
        mat = np.column_stack([image1_intensities, image2_intensities])
        mat= mat.astype('float32')
        mat -= mat.mean(axis=0)
        fact = len(data_points) - 1
        co_var= np.dot(mat.T, mat.conj()) / fact
        co_var_inv = np.linalg.inv(co_var)
        det = np.linalg.det(co_var)
        print("Covariance matrix obtained is {}".format(co_var))


        #Visualization of ellipse from co variance matrix
        multiplier = stats.chi2.ppf(q=threshold, df=2)
        eigen_values, eigen_vectors = np.linalg.eigh(co_var)
        minor_axis, major_axis = 4 * np.sqrt(eigen_values[:, None] * multiplier)
        theta = np.degrees(np.arctan2(*eigen_vectors[::-1, 0]))
        ellipse = Ellipse(xy=(np.mean(image1_intensities), np.mean(image2_intensities)),
                      width=minor_axis, height=major_axis,
                      angle=theta, color='black')
        ellipse.set_facecolor('none')
        gaussian_fitting=self.get_image(image1_intensities,image2_intensities,ellipse)

        outliers=[]
        image_difference=[]

        #Based on the probability obtained through the formula, comparing against a threshold
        factor1 = 1 / (2 * (22 / 7) * np.sqrt(det))
        norm_factor = 250000
        for i in range(len(data_points)):
            mat_tr = []
            mat_tr.append(data_points[i][0] - x_mean)
            mat_tr.append(data_points[i][1] - y_mean)
            trs_mat = np.reshape(mat_tr, (1, 2))
            mat_original = np.transpose(trs_mat)

            mat_mul_value = np.matmul(trs_mat, co_var_inv)
            mat_mul_value1 = np.matmul(mat_mul_value, mat_original)
            factor2 = math.exp(-(0.5 * (mat_mul_value1)))
            prob_value = norm_factor * factor1 * factor2
            if (prob_value < threshold):
                outliers.append(data_points[i])
                image_difference.append(255)
            else:
                image_difference.append(0)

        outliers_x = []
        outliers_y = []

        for i in range(len(outliers)):
            outliers_x.append(outliers[i][0])
            outliers_y.append(outliers[i][1])

        thresholded_gaussian =  self.get_image(outliers_x,outliers_y,'None')
        segmented_gaussian = np.reshape(image_difference, (image2.shape[0], image2.shape[1]))
        kernel = np.ones((2, 2), np.uint8)
        segmented_gaussian = cv2.erode((segmented_gaussian * 1.0).astype(np.float32), kernel)
        outlier_percentage = (len(outliers) / len(data_points)) * 100
        print("GA")
        return (gaussian_fitting,  thresholded_gaussian, segmented_gaussian)

    def get_image(self,x,y,figure):
        fig = plt.figure()
        plot = fig.add_subplot(111)
        if(figure!='None'):
            plot.add_artist(figure)
        plot.scatter(x, y)
        fig.canvas.draw()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image