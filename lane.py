import numpy as np

def is_sane(left_plot_x, right_plot_x):
    d_lower = right_plot_x[0] - left_plot_x[0]
    d_upper = right_plot_x[-1] - left_plot_x[-1]
    return d_lower >= 290 and d_lower <= 800 and d_upper >= 490 and d_upper <= 700


ym_per_pix = 30/720.0 # meters per pixel in y dimension
xm_per_pix = 3.7/700.0 # meters per pixel in x dimension

def find_curvature(leftx, rightx, ploty):  
    # Evaluation point  
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    #if left_fit_cr[0] * right_fit_cr[0] < 0: # one line turns left, the other turns right
    #    return None

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    curverad = (left_curverad + right_curverad) / 2

    #if curverad > 10000:
    #    return None

    return curverad

def find_relative_car_position(lane_center, image_center):
    return (image_center - lane_center) * xm_per_pix

class Lane:
    MAX_AGE = 50
    TEXT_CHANGE_AFTER_FRAMES = 10

    def __init__(self, image_shape):
        self.age = Lane.MAX_AGE
        self.frame = 0

        self.image_shape = image_shape
        self.image_center_x = image_shape[1] / 2.0
        
        self.points = None
        self.left_poly = None
        self.right_poly = None
        
        self.center_x = None
        self.radius_of_curvature_m = None # meters
        self.relative_car_position_m = None # meters

        self.radius_of_curvature_for_display_m = None
        self.relative_car_position_for_display_m = None

    def is_up_to_date(self):
        return self.age == 1

    def is_valid(self):
        return self.age < Lane.MAX_AGE

    def update(self, left_plot_x, right_plot_x, plot_y, left_poly, right_poly):
        if is_sane(left_plot_x, right_plot_x):
            points_left = np.array([np.transpose(np.vstack([left_plot_x, plot_y]))])
            points_right = np.array([np.flipud(np.transpose(np.vstack([right_plot_x, plot_y])))])
            points = np.hstack((points_left, points_right))

            if self.is_valid():
                a = 0.8 ** self.age
                self.points = a * self.points + (1-a) * points
                self.left_poly = a * self.left_poly + (1-a) * left_poly
                self.right_poly = a * self.right_poly + (1-a) * right_poly
            else:
                self.points = points
                self.left_poly = left_poly
                self.right_poly = right_poly

            self.radius_of_curvature_m = find_curvature(left_plot_x, right_plot_x, plot_y)
            self.center_x = (left_plot_x[-1] + right_plot_x[-1]) / 2.0
            self.relative_car_position_m = find_relative_car_position(self.center_x, self.image_center_x)
            if self.frame == 0:
                self.radius_of_curvature_for_display_m = self.radius_of_curvature_m
                self.relative_car_position_for_display_m = self.relative_car_position_m
            self.frame = (self.frame + 1) % Lane.TEXT_CHANGE_AFTER_FRAMES
            self.age = 1
        else:
            self.update_insane()

    def update_insane(self):
        # The min() is just to avoid overflow
        self.age = min(self.age + 1, Lane.MAX_AGE)
