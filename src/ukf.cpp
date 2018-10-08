#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // State dimension
    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = 7;

    // Radar measurement dimension
    n_z_rad_ = 3;

    // Lidar measurement dimension
    n_z_las_ = 2;

    lambda_ = 3 - n_aug_;

    // initial state vector
    x_ = VectorXd(n_x_);

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);

    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    previous_timestamp_ = 0;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1;

    //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

    weights_ = VectorXd(2 * n_aug_ + 1);
    double weight_0 = lambda_ / (lambda_ + n_aug_);

    weights_(0) = weight_0;
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
        double weight = 0.5 / (n_aug_ + lambda_);
        weights_(i) = weight;
    }

    NIS_radar_ = 0.0;

    NIS_laser_ = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if (!is_initialized_) {
        P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;

        if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER) {

            // Initialize state
            x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;

        } else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR) {

            // Convert measurement coordinates into state coordinates
            double rho = meas_package.raw_measurements_(0);
            double phi = meas_package.raw_measurements_(1);
            double rhod = meas_package.raw_measurements_(2);

            double px = rho * cos(phi);
            double py = rho * sin(phi);
            double vx = rhod * cos(phi);
            double vy = rhod * sin(phi);
            double v = sqrt(pow(vx, 2) + pow(vy, 2));

            x_ << px, py, v, 0, 0;

        }

        previous_timestamp_ = meas_package.timestamp_;
        is_initialized_ = true;
    }

    double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = meas_package.timestamp_;

    Prediction(delta_t);

    if (meas_package.sensor_type_ == MeasurementPackage::SensorType::LASER && use_laser_) {
        VectorXd z_pred = VectorXd::Zero(n_z_las_);
        MatrixXd S = MatrixXd::Zero(n_z_las_,n_z_las_);
        MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_las_);

        PredictLidarMeasurement(z_pred, S, Tc);
        UpdateLidar(meas_package, z_pred, S, Tc);
    } else if (meas_package.sensor_type_ == MeasurementPackage::SensorType::RADAR && use_radar_) {
        VectorXd z_pred = VectorXd::Zero(n_z_rad_);
        MatrixXd S = MatrixXd::Zero(n_z_rad_,n_z_rad_);
        MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_rad_);

        PredictRadarMeasurement(z_pred, S, Tc);
        UpdateRadar(meas_package, z_pred, S, Tc);
    }
}

/**
 * Generates sigma points for the augmented state
 * @param Xsig_out The output state augmentation matrix
 */
void UKF::AugmentedSigmaPoints(MatrixXd& Xsig_out) {
    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_ * std_a_;
    P_aug(6,6) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i < n_aug_; i++)
    {
        Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
    }

    Xsig_out = Xsig_aug;
}

/**
 * Calculates sigma points for predicted state
 * @param Xsig_aug Input state augmentation matrix
 * @param delta_t Timestep
 */
void UKF::SigmaPointPrediction(MatrixXd& Xsig_aug, const double delta_t) {
    //predict sigma points
    for (int i = 0; i< 2 * n_aug_ + 1; i++)
    {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }
}

/**
 * Predicts mean state vector and covariance matrix
 */
void UKF::PredictMeanAndCovariance() {
    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        //angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2.0*M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2.0*M_PI;

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    }
}

/**
 * Predicts radar measurements
 * @param z_out Output z measurement prediction
 * @param S_out Innovation covariance matrix
 * @param T_out Cross correlation matrix
 */
void UKF::PredictRadarMeasurement(VectorXd& z_out, MatrixXd& S_out, MatrixXd& T_out) {
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd::Zero(n_z_rad_, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
        Zsig(1,i) = atan2(p_y,p_x);                                 //phi
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z_rad_);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //innovation covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z_rad_,n_z_rad_);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2.0*M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2.0*M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd::Zero(n_z_rad_,n_z_rad_);
    R <<    std_radr_*std_radr_, 0, 0,
            0, std_radphi_*std_radphi_, 0,
            0, 0,std_radrd_*std_radrd_;
    S = S + R;

    z_out = z_pred;
    S_out = S;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_rad_);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2.0*M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2.0*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        //angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2.0*M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2.0*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    T_out = Tc;
}

/**
 *
 * @param z_out Output z measurement prediction
 * @param S_out Innovation covariance matrix
 * @param T_out Cross correlation matrix
 */
void UKF::PredictLidarMeasurement(VectorXd& z_out, MatrixXd& S_out, MatrixXd& T_out) {
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd::Zero(n_z_las_, 2 * n_aug_ + 1);

    //// extract values for better readibility
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        Zsig(0,i) = Xsig_pred_(0,i);
        Zsig(1,i) = Xsig_pred_(1,i);
    }

    VectorXd z_pred = VectorXd::Zero(n_z_las_);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //innovation covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z_las_,n_z_las_);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2.0*M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2.0*M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd::Zero(n_z_las_,n_z_las_);
    R <<    std_laspx_*std_laspx_, 0,
            0, std_laspy_*std_laspy_;
    S = S + R;

    z_out = z_pred;
    S_out = S;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_las_);
    Tc.fill(0.0);

    //calculate cross correlation matrix

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2.0*M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2.0*M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        //angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2.0*M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2.0*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    T_out = Tc;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

    AugmentedSigmaPoints(Xsig_aug);
    SigmaPointPrediction(Xsig_aug, delta_t);
    PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package, VectorXd& z_pred, MatrixXd& S, MatrixXd& Tc) {
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    VectorXd z = VectorXd::Zero(n_z_las_);
    z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0*M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0*M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    // Calculate radar NIS and find average difference from 95% line
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package, VectorXd& z_pred, MatrixXd& S, MatrixXd& Tc) {
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    VectorXd z = VectorXd::Zero(n_z_rad_);
    z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.0*M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0*M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    // Calculate radar NIS and find average difference from 95% line
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
