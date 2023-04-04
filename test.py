from scipy import io
import numpy as np
import os

import matplotlib.pyplot as plt
# import cv2
from math import pi
import math
# from skvideo.io import FFmpegWriter
# import cPickle as pickle

#######################################################################
#######################################################################

Dataset = 1



#######################################################################
#######################################################################

def cylindrical_projection(src, rot, dst=None):
    nRows, nCols, c = src.shape
    y = np.linspace(0, nCols - 1, nCols)
    z = np.linspace(0, nRows - 1, nRows)
    yy, zz = np.meshgrid(y, z)
    yy = yy.reshape((1, nCols * nRows))
    zz = zz.reshape((1, nCols * nRows))
    rgb = np.transpose(src, (2, 0, 1)).reshape((3, nCols * nRows)).T

    yy = (nCols / 2) - yy
    zz = (nRows / 2) - zz

    hFOV = 60 * (pi / 180)
    yFactor = hFOV / nCols
    vFOV = 45 * (pi / 180)
    zFactor = vFOV / nRows
    longitude = yy * yFactor
    latitude = zz * zFactor

    rho = 1
    X = rho * np.cos(latitude) * np.cos(longitude)
    Y = rho * np.cos(latitude) * np.sin(longitude)
    Z = rho * np.sin(latitude)
    XYZ = np.vstack((X, Y, Z))

    wXYZ = rot.dot(XYZ)

    wX = wXYZ[0, :]
    wY = wXYZ[1, :]
    wZ = wXYZ[2, :]
    longitude = np.arctan2(wY, wX)
    latitude = np.arctan2(wZ, np.sqrt(wX ** 2 + wY ** 2))

    cylHeight = ((1 / zFactor) * (-np.tan(latitude) + pi / 2)).astype(np.uint32)
    cylAngle = ((1 / yFactor) * (-longitude + pi)).astype(np.uint32)

    if dst is None:
        dst = np.zeros((int(pi / zFactor), int(2 * pi / yFactor), 3)).astype(np.uint8)

    try:
        dst[cylHeight, cylAngle, :] = rgb

    except IndexError:
        cylHeight[cylHeight > dst.shape[0] - 1] = dst.shape[0] - 1
        dst[cylHeight, cylAngle, :] = rgb

    return dst


def gaussian_update(qt, ut, P, Q):
    qu = vec2quat(ut)

    tmp = np.matrix(np.zeros([4, 6]))
    L = np.linalg.cholesky(P + Q)
    n, m = np.shape(P)
    left_vec = L * np.sqrt(2 * n)
    right_vec = -L * np.sqrt(2 * n)
    new_vec = np.hstack((left_vec, right_vec))
    nr, nc = np.shape(new_vec)

    v = np.matrix(np.zeros([3, 6]))
    for i in range(0, nc):
        temp = vec2quat(new_vec[:, i])
        tmp[:, i] = np.transpose(multiply_quaternions(temp, qt))

    sigma_points = np.transpose(tmp)
    motion_sig = np.zeros(np.shape(sigma_points))
    for i in range(0, 6):
        motion_sig[i] = multiply_quaternions(sigma_points[i], qu)
    next_qt, error = quat_average(motion_sig, qt)
    nr, nc = np.shape(error)
    next_cov = np.zeros([nc, nc])
    for i in range(0, nr):
        temp_cov = np.transpose(error[i]) * error[i]
        next_cov += temp_cov

    next_cov = next_cov / 12
    return next_qt, next_cov, sigma_points, error


def sigma_update(sigma_points, g, R):
    new_sigma = np.zeros(np.shape(sigma_points))
    z = np.zeros([np.shape(sigma_points)[0] - 1, np.shape(sigma_points)[1]])
    for i in range(0, np.shape(sigma_points)[0]):
        new_sigma[i] = multiply_quaternions(multiply_quaternions(inverse_quaternion(sigma_points[i]), g),
                                            sigma_points[i])

    z = new_sigma[:, 1:]
    z_mean = np.mean(z, 0)
    return z, z_mean


def calcpzz(z, z_mean):
    temp = np.matrix(z - z_mean)
    pzz = np.zeros([np.shape(z)[1], np.shape(z)[1]])
    for i in range(0, np.shape(temp)[0]):
        pzz_temp = np.transpose(temp[i]) * temp[i]
        pzz += pzz_temp

    return pzz / 12.0


def calcpxz(error, z, z_mean):
    temp = np.matrix(z - z_mean)
    pxz = np.zeros([np.shape(z)[1], np.shape(z)[1]])
    for i in range(0, np.shape(error)[0]):
        pxz_temp = np.transpose(error[i]) * temp[i]
        pxz += pxz_temp

    return pxz / 12.0





def quat_average(q, q0):
    q = np.matrix(q)
    qt = q0
    nr, nc = np.shape(q)
    qe = np.matrix(np.zeros([nr, 4]))
    ev = np.matrix(np.zeros([nr, 3]))
    pi = 22.0 / 7
    epsilon = 0.0001
    temp = np.zeros([1, 4])
    for t in range(1000):
        for i in range(0, nr, 1):
            q[i] = normalize_quaternion(q[i])
            qe[i] = multiply_quaternions(q[i], inverse_quaternion(qt))
            qs = qe[i, 0]
            qv = qe[i, 1:4]
            if np.round(norm_quaternion(qv), 8) == 0:
                if np.round(norm_quaternion(qe[i]), 8) == 0:
                    ev[i] = np.matrix([0, 0, 0])
                else:
                    ev[i] = np.matrix([0, 0, 0])
            if np.round(norm_quaternion(qv), 8) != 0:
                if np.round(norm_quaternion(qe[i]), 8) == 0:
                    ev[i] = np.matrix([0, 0, 0])
                else:
                    temp[0, 0] = np.log(norm_quaternion(qe[i]))
                    temp[0, 1:4] = np.dot((qv / norm_quaternion(qv)), math.acos(qs / norm_quaternion(qe[i])))
                    ev[i] = 2 * temp[0, 1:4]
                    ev[i] = ((-np.pi + (np.mod((norm_quaternion(ev[i]) + np.pi), (2 * np.pi)))) / norm_quaternion(
                        ev[i])) * ev[i]
        e = np.transpose(np.mean(ev, 0))
        temp2 = np.array(np.zeros([4, 1]))
        temp2[0] = 0
        temp2[1:4] = e / 2.0
        temp2 += 0.00001 * np.ones(temp2.shape)
        qt = multiply_quaternions(exp_quaternion(np.transpose(temp2)), qt)

        if norm_quaternion(e) < epsilon:
            return qt, ev


def multiply_quaternions(q, r):
    t = np.empty(([1, 4]))
    t[:, 0] = r[:, 0] * q[:, 0] - r[:, 1] * q[:, 1] - r[:, 2] * q[:, 2] - r[:, 3] * q[:, 3]
    t[:, 1] = (r[:, 0] * q[:, 1] + r[:, 1] * q[:, 0] - r[:, 2] * q[:, 3] + r[:, 3] * q[:, 2])
    t[:, 2] = (r[:, 0] * q[:, 2] + r[:, 1] * q[:, 3] + r[:, 2] * q[:, 0] - r[:, 3] * q[:, 1])
    t[:, 3] = (r[:, 0] * q[:, 3] - r[:, 1] * q[:, 2] + r[:, 2] * q[:, 1] + r[:, 3] * q[:, 0])

    return t


def conjugate_quaternion(q):
    t = np.empty([4, 1])
    t[0] = q[0]
    t[1] = -q[1]
    t[2] = -q[2]
    t[3] = -q[3]

    return t


def divide_quaternions(q, r):
    t = np.empty([4, 1])
    t[0] = ((r[0] * q[0]) + (r[1] * q[1]) + (r[2] * q[2]) + (r[3] * q[3])) / (
                (r[0] ** 2) + (r[1] ** 2) + (r[2] ** 2) + (r[3] ** 2))
    t[1] = (r[0] * q[1] - (r[1] * q[0]) - (r[2] * q[3]) + (r[3] * q[2])) / (
                r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)
    t[2] = (r[0] * q[2] + r[1] * q[3] - (r[2] * q[0]) - (r[3] * q[1])) / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)
    t[3] = (r[0] * q[3] - (r[1] * q[2]) + r[2] * q[1] - (r[3] * q[0])) / (r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)

    return t


def inverse_quaternion(q):
    t = np.empty([4, 1])
    t[0] = q[:, 0]
    t[1] = -q[:, 1]
    t[2] = -q[:, 2]
    t[3] = -q[:, 3]
    t=normalize_quaternion(t)

    t = np.transpose(t)

    return t


def norm_quaternion(q):
    t = np.sqrt(np.sum(np.power(q, 2)))
    return t


def normalize_quaternion(q):
    return q / norm_quaternion(q)


def rotate_vector_by_quaternion(q, v):
    v_rotated = []
    v_rotated = np.matrix(
        [[(1 - 2 * (q[2] ^ 2) - 2 * (q[3] ^ 2)), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * ((q[1] * q[3]) - (q[0] * q[2]))],
         [2 * (q[1] * q[2] - q[0] * q[3]), (1 - 2 * (q[1] ^ 2) - 2 * (q[3] ^ 2)), 2 * ((q[2] * q[3]) + (q[0] * q[1]))],
         [2 * (q[1] * q[3] + q[0] * q[2]), 2 * ((q[2] * q[3]) - (q[0] * q[1])),
          (1 - 2 * (q[1] ^ 2) - 2 * (q[2] ^ 2))]]) * v
    return v_rotated


def quat2rot(q):
    q = normalize_quaternion(q)

   # print(q)
    qhat = np.zeros([3, 3])
    qhat[0, 1] = -q[:, 3]
    qhat[0, 2] = q[:, 2]
    qhat[1, 2] = -q[:, 1]
    qhat[1, 0] = q[:, 3]
    qhat[2, 0] = -q[:, 2]
    qhat[2, 1] = q[:, 1]

    R = np.identity(3) + 2 * np.dot(qhat, qhat) + 2 * np.array(q[:, 0]) * qhat
    return R


def rot2euler(R):
    phi = -math.asin(R[1, 2])
    theta = -math.atan2(-R[0, 2] / math.cos(phi), R[2, 2] / math.cos(phi))
    psi = -math.atan2(-R[1, 0] / math.cos(phi), R[1, 1] / math.cos(phi))

    return phi, theta, psi


def rot2quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2];

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S

    elif ((R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2])):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S

    elif (R[1, 1] > R[2, 2]):
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = [[qw], [qx], [qy], [qz]]
    temp = np.sign(qw)
    q = np.multiply(q, temp)
    return q


def vec2quat(r):
    r = r / 2.0
    q = np.matrix(np.zeros([4, 1]))
    q[0] = math.cos(np.linalg.norm(r))
    if np.linalg.norm(r) == 0:
        temp = np.transpose(np.matrix([0, 0, 0]))
    else:
        temp = np.transpose(np.matrix((r / np.linalg.norm(r)) * (math.sin(np.linalg.norm(r)))))
    q[1:4] = temp
    q = np.transpose(q)
    return q


def quat2vec(q):
    qs = q[:, 0]
    qv = q[:, 1:4]
    if np.linalg.norm(qv) == 0:
        v = np.transpose(np.matrix([0, 0, 0]))
    else:
        v = 2 * ((qv / np.linalg.norm(qv)) * math.acos(qs / np.linalg.norm(q)))
    return v


def log_quaternion(qe):
    qe = np.transpose(qe)
    qs = qe[0]
    qv = qe[1:4]
    log_q = np.zeros(np.shape(qe))

    log_q[0] = np.log(norm_quaternion(qe))
    log_q[1:4] = np.dot(qv / norm_quaternion(qv), math.acos(qs / norm_quaternion(qe)))
    return log_q


def exp_quaternion(q):
    q = np.transpose(q)
    qs = q[0]
    qv = q[1:4]
    exp_q = np.zeros(np.shape(q))

    exp_q[0] = math.cos((norm_quaternion(qv)))
    exp_q[1:4] = np.dot(normalize_quaternion(qv), math.sin(norm_quaternion(qv)))
    return np.transpose(math.exp(qs) * exp_q)

########################################################################
########################################################################

# Data Load and timestamp match

imu = io.loadmat('imuRaw' + str(Dataset) + '.mat')
imu_vals = imu['vals']
imu_vals = np.transpose(imu_vals)
imu_ts = imu['ts']
yts = imu_ts
imu_ts = np.transpose(imu_ts)

Vref = 3300

acc_x = -np.array(imu_vals[:, 0])
acc_y = -np.array(imu_vals[:, 1])
acc_z = np.array(imu_vals[:, 2])
acc = [acc_x, acc_y, acc_z]

acc = np.array(acc)
acc = np.transpose(acc)
acc_sensitivity = 330.0
acc_scale_factor = Vref / 1023.0 / acc_sensitivity
acc_bias = acc[0] - (np.array([0, 0, 1]) / acc_scale_factor)
acc_val = acc * acc_scale_factor
acc_val = acc_val - (acc_bias) * acc_scale_factor

gyro_x = np.array(imu_vals[:, 4])
gyro_y = np.array(imu_vals[:, 5])
gyro_z = np.array(imu_vals[:, 3])

gyro = [gyro_x, gyro_y, gyro_z]
gyro = np.array(gyro)
gyro = np.transpose(gyro)
gyro_bias = gyro[0]
gyro_sensitivity = 3.33
gyro_scale_factor = Vref / 1023 / gyro_sensitivity
gyro_val = gyro * gyro_scale_factor
gyro_val = (np.array(gyro_val) - (gyro_bias * gyro_scale_factor)) * (np.pi / 180)

if os.path.exists("viconRot" + str(Dataset) + ".mat"):

    vicon = io.loadmat("viconRot" + str(Dataset) + ".mat")

    vicon_vals = vicon['rots']
    vicon_ts = vicon['ts']

    vicon_phi = np.zeros([np.shape(vicon_vals)[2], 1])
    vicon_theta = np.zeros([np.shape(vicon_vals)[2], 1])
    vicon_psi = np.zeros([np.shape(vicon_vals)[2], 1])
    for i in range(np.shape(vicon_vals)[2]):
        R = vicon_vals[:, :, i]
        vicon_phi[i], vicon_theta[i], vicon_psi[i] = rot2euler(R)

else:
    est = True

########################################################################
########################################################################

P = 0.00001 * np.identity(3)
Q = 0.00001 * np.identity(3)
R = 0.0001 * np.identity(3)
q0 = np.matrix([1, 0, 0, 0])
qt = np.matrix([1, 0, 0, 0])
ut = gyro_val[0]
g = np.matrix([0, 0, 0, 1])
t = imu_ts.shape[0]
R_calc = np.zeros((3, 3, np.shape(gyro_val)[0]))

# UKF

for i in range(0, np.shape(gyro_val)[0]):

    if i == 0:
        ut = gyro_val[i] * imu_ts[0]
        predicted_q = q0
    else:
        ut = gyro_val[i] * (imu_ts[i] - imu_ts[i - 1])

    next_q, next_cov, sigma_points, error = gaussian_update(qt, ut, P, Q)

    z, z_mean = sigma_update(sigma_points, g, R)
    z = np.matrix(z)
    z_mean = np.matrix(z_mean)

    pzz = calcpzz(z, z_mean)
    pvv = pzz + R
    pxz = calcpxz(error, z, z_mean)

    K = np.dot(pxz, np.linalg.inv(pvv))

    I = np.transpose(acc_val[i] - z_mean)
    KI = vec2quat(np.transpose(K * I))
    qt = np.matrix(np.empty([1, 4]))
    qt = multiply_quaternions(KI, next_q)
    P = next_cov - np.dot(np.dot(K, pvv), np.transpose(K))

    predicted_q = np.vstack((predicted_q, qt))

    R_calc[:, :, i] = quat2rot(qt)





phi = np.zeros([np.shape(predicted_q)[0], 1])
theta = np.zeros([np.shape(predicted_q)[0], 1])
psi = np.zeros([np.shape(predicted_q)[0], 1])

for i in range(np.shape(predicted_q)[0]):
    R = quat2rot(predicted_q[i])
    phi[i], theta[i], psi[i] = rot2euler(R)


plt.figure(1)
plt.subplot(311)
plt.plot(vicon_phi, 'b', phi, 'r')
plt.ylabel('Roll')
plt.subplot(312)
plt.plot(vicon_theta, 'b', theta, 'r')
plt.ylabel('Pitch')
plt.subplot(313)
plt.plot(vicon_psi, 'b', psi, 'r')
plt.ylabel('Yaw')

plt.show()


