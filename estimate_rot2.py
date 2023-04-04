import os
import numpy as np

from scipy import io


import math

def quat_sigmapoints(qt,P, Q):
    r = P.shape[0]
    buff = np.zeros((2*r,4))

    S = np.linalg.cholesky(P + Q)
    # print(S)

    # print(n,m)
    # column vectors+or -sqrt2N
    col_1 = S * np.sqrt(2 * r)

    col_2 = -S * np.sqrt(2 * r)
    W_i = np.hstack((col_1, col_2))
    # print(W_i)


    for i in range(2*r):
        buff2 = vec2quat(W_i[:, i])
        # print(qt.shape, buff2.shape)
        buff[i,: ] = multiply_quaternions(qt, buff2)

    buff=np.vstack((qt,buff))

    return buff
def transform_sigmapoints(tmp,gyro,dt):
    # qu = vec2quat(ut)
    a=tmp.shape[0]
    # sigma_points = np.transpose(tmp)
    qu=vec2quat(gyro*dt)
    Y_sig = np.zeros((a,4))
    # print(sigma_points.shape,motion_sig.shape)
    for i in range(0, a):
        q=tmp[i,:]
        Y_sig[i,:] = multiply_quaternions(q, qu)
    return Y_sig
    # print(motion_sig.shape)

def predict(Y_sig,q):

    n=Y_sig.shape[0]

    qt_update, error = comp_mean(Y_sig, q)


    cov_update = np.zeros((3, 3))
    for i in range(0, n):
        cov_update += np.outer(error[i,:],error[i,:])

    cov_update = cov_update / n
    return qt_update, cov_update, error


def comp_mean(q_i, q_t):
    n=q_i.shape[0]


    # qw = np.zeros([1, 4]
    for a in range(1000):
        error_vec=np.zeros((n,3))

        for i in range(0, q_i.shape[0]):
            # q_i[i] = normalize(q_i[i])

            q_ierr = normalize(multiply_quaternions(q_i[i,:], inverse_quaternion(q_t)))
            v_e=quat2vec(q_ierr)

            vi_norm=np.linalg.norm(v_e)

            if vi_norm==0:
                error_vec[i,:]=np.zeros(3)
            else:
                error_vec[i,:]=(-np.pi+np.mod(vi_norm+np.pi,2*np.pi))/vi_norm*v_e
        error=np.mean(error_vec,axis=0)
        q_t = normalize(multiply_quaternions(vec2quat(error), q_t))

        if np.linalg.norm(error) < 0.001:
            break

    return q_t, error_vec

def update_points(sigma_points):
    n=sigma_points.shape[0]
    quat_G = np.array([0,0,0,1])
    z = np.zeros((n,3))


    for i in range(0,sigma_points.shape[0]):
        q=sigma_points[i,:]
        z[i,:] = multiply_quaternions(multiply_quaternions(inverse_quaternion(q), quat_G),
                                            q)[1:]


    z_mean = np.mean(z, axis=0)
    z_mean =z_mean/np.linalg.norm(z_mean)
    return z, z_mean


def measurementestimate_covariance(sigma_points,z, z_mean):
    n=sigma_points.shape[0]
    z_error = z - z_mean
    pzz = np.zeros((3,3))

    for i in range(0, n):
        pzz += np.outer(z_error[i,:],z_error[i,:])

    pzz/=n

    return pzz


def crosscorrelation(sigma_points,error, z, z_mean):
    n = sigma_points.shape[0]
    z_error = z - z_mean
    pxz = np.zeros((3,3))
    for i in range(0, n):
        pxz += np.outer(error[i,:],z_error[i,:])

    pxz/=n

    return pxz

def multiply_quaternions(p, q):
    p0 = p[0]
    pv = p[1:]
    q0 = q[0]
    qv = q[1:]
    # print(pv.shape,qv.shape)
    r0 = p0 * q0 - np.dot(pv, qv)
    rv = p0 * qv + q0 * pv + np.cross(pv, qv)

    return np.array([r0, rv[0], rv[1], rv[2]])
def normalize(q):
    return q / np.linalg.norm(q)
def inverse_quaternion(q):
    q_conj = quat_conjugate(q)
    q_norm = np.linalg.norm(q)
    return q_conj / (q_norm ** 2)
def quat_conjugate(q):
    q_conj = np.copy(q)
    q_conj[1:] *= -1
    return q_conj

def quat2rot(q):
    q = normalize(q)
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

def quat2vec(q):
    # log mapping
    r = (2*quat_log(q))[1:]
    return r
def quat_log(q):
    qnorm = np.linalg.norm(q)
    q0 = q[0]
    qv = q[1:]
    qvnorm = np.linalg.norm(qv)

    z0 = np.log(qnorm)
    if qvnorm == 0:
        zv = np.zeros(3)
    else:
        zv = (qv / qvnorm) * np.arccos(q0 / qnorm)
    return np.array([z0, zv[0], zv[1], zv[2]])
def vec2quat(vec):
    # exp mapping
    r = vec / 2
    q = quat_exp([0, r[0], r[1], r[2]])
    return q
def quat_exp(q):
    q0 = q[0]
    qv = q[1:]
    qvnorm = np.linalg.norm(qv)

    z0 = np.exp(q0) * np.cos(qvnorm)
    if qvnorm == 0:
        zv = np.zeros(3)
    else:
        zv = np.exp(q0) * (qv / qvnorm) * np.sin(qvnorm)
    return np.array([z0, zv[0], zv[1], zv[2]])

def euler_angles(q):
    phi = math.atan2(2*(q[0]*q[1]+q[2]*q[3]), \
    1 - 2*(q[1]**2 + q[2]**2))
    theta = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
    psi = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), \
    1 - 2*(q[2]**2 + q[3]**2))
    return np.array([phi, theta, psi])
def estimate_rot(data_num):


    imu = io.loadmat(os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat"))
    # imu = io.loadmat('imuRaw' + str(data_num) + '.mat')


    imu_vals=np.array(imu['vals'])

    imu_ts = np.array(imu['ts']).T


    # aacc = acc.reshape(accel.shape[1],accel.shape(0))
    acc_x = -imu_vals[0,: ]
    acc_y = -imu_vals[1,: ]
    acc_z = imu_vals[2,: ]

    acc = np.array([acc_x, acc_y, acc_z]).T

    Vref = 3300
    acc_sensitivity = 330
    acc_scale_factor = Vref / 1023.0 / acc_sensitivity

    acc_bias = np.mean(acc[:10], axis = 0) - (np.array([0,0,1])/acc_scale_factor)

    acc_afterC = (acc-acc_bias)*acc_scale_factor




    gyro_x = np.array(imu_vals[4,: ])
    gyro_y = np.array(imu_vals[5,: ])
    gyro_z = np.array(imu_vals[3,: ])

    gyro = np.array([gyro_x, gyro_y, gyro_z]).T


    gyro_bias = np.mean(gyro[:10],axis=0)
    # print(gyro_bias,gyro[0])
    gyro_sensitivity = 3.33
    gyro_scale_factor = Vref / 1023 / gyro_sensitivity
    # gyro_val = gyro * gyro_scale_factor
    gyro_afterC = (gyro-gyro_bias)*gyro_scale_factor*(np.pi/180)
    # print(gyro_afterC.shape)




    P = 0.1 * np.identity(3)
    # print(P)
    Q = 2 * np.identity(3)
    R = 2 * np.identity(3)

    qk = np.array([1,0,0,0])



    ts=imu_ts.shape[0]
    print(ts)
    # R_calc = np.zeros((ts,3))
    Rcal = np.zeros((ts, 3))



    for i in range(0, ts):
        acc=acc_afterC[i]
        # print(acc)

        gyro=gyro_afterC[i]
        # print(gyro)



        s1_points = quat_sigmapoints(qk, P, Q)


        if i == ts-1: # last iter
            dt = np.mean(imu_ts[-10:] - imu_ts[-11:-1])
        else:
            dt = imu_ts[i+1] - imu_ts[i]

        sigma_points = transform_sigmapoints(s1_points, gyro, dt)



        next_q, next_cov, error = predict(sigma_points, qk)


        z, z_mean = update_points(sigma_points)
        pzz = measurementestimate_covariance(sigma_points, z, z_mean)
        pvv = pzz + R
        pxz = crosscorrelation(sigma_points, error, z, z_mean)

        acc /= np.linalg.norm(acc)
        I = acc - z_mean
        K = np.dot(pxz, np.linalg.inv(pvv))

        gain = vec2quat(K.dot(I))



        q_u = multiply_quaternions(gain, next_q)
        cov_update = next_cov - K.dot(pvv).dot(K.T)
        P = cov_update
        qk = q_u



        Rcal[i, :] = euler_angles(qk)

    print(Rcal[:, 1].shape)
    return Rcal[:, 0],Rcal[:, 1],Rcal[:, 2]
#   
#
#
#
#
# estimate_rot(3)