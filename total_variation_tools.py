import numpy as np
import scipy.sparse
import time
import cv2


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def make_nabla(M, N):
    row = np.arange(0, M*N)
    dat = np.ones(M*N)
    col = np.arange(0, M*N).reshape(M, N)
    col_xp = np.block([[col[:, 1:], col[:, -1:]]])
    col_yp = np.block([[col[1:, :]], [col[-1:, :]]])

    nabla_x = scipy.sparse.coo_matrix(
        (dat, (row, col_xp.flatten())), shape=(M*N, M*N)
    ) - scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(M*N, M*N))

    nabla_y = scipy.sparse.coo_matrix(
        (dat, (row, col_yp.flatten())), shape=(M*N, M*N)
    ) - scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(M*N, M*N))

    nabla = scipy.sparse.vstack([nabla_x, nabla_y])

    return nabla


def make_nabla_x(M, N):
    # Version with the constant y component, good for images with horizontal scratches
    row = np.arange(0, M*N)
    dat = np.ones(M*N)
    col = np.arange(0, M*N).reshape(M, N)
    col_xp = np.block([[col[:, 1:], col[:, -1:]]])
    col_yp = np.block([[col[1:, :]], [col[-1:, :]]])

    nabla_x = scipy.sparse.coo_matrix(
        (dat, (row, col_xp.flatten())), shape=(M*N, M*N)
    ) - scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(M*N, M*N))

    nabla_y = scipy.sparse.coo_matrix(
        (np.zeros(M*N), (row, col_yp.flatten())), shape=(M*N, M*N)
    )  # MY CONSTANT COMPONENT

    nabla = scipy.sparse.vstack([nabla_x, nabla_y])

    return nabla


def make_nabla_y(M, N):
    # Version with the constant x component
    row = np.arange(0, M*N)
    dat = np.ones(M*N)
    col = np.arange(0, M*N).reshape(M, N)
    col_xp = np.block([[col[:, 1:], col[:, -1:]]])
    col_yp = np.block([[col[1:, :]], [col[-1:, :]]])

    nabla_x = scipy.sparse.coo_matrix(
        (np.zeros(M*N), (row, col_xp.flatten())), shape=(M*N, M*N)
    )  # MY CONSTANT COMPONENT
    nabla_y = scipy.sparse.coo_matrix(
        (dat, (row, col_yp.flatten())), shape=(M*N, M*N)
    ) - scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(M*N, M*N))

    nabla = scipy.sparse.vstack([nabla_x, nabla_y])

    return nabla


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def l1TV_ROF(
    image, iterations, nabla_comp, algo, lam, tau, alpha, err, gif=False, overrelax=True):
    """
    Applies different denoising algorithms to an image.
    For now l1_TV and Huber-ROF.
    |----------------------------
    Parameters:
    - image - input image. Grayscale at least for now.
    - iterations - number of iterations.
    - nabla_comp - choose either 'both', 'x', 'y' to select the nonzero components
                   of the operator.
    - algo - 1: l1-TV, 2: Huber-ROF
    - lam - parameter lambda in TV.
    - tau -    //     tau      // .
    - alpha - parameter in Huber-ROF. Ignore if TV is chosen.
    - err: RMSE threshold for convergence. Set to 0 to turn execute a 'iteration' number of iterations.
    - gif - boolean to choose whether to save or not the imgs for the future gif.
    - overrelax : boolean, turn on/off the overrelaxation. Default: True. False for (testing) in case of terraces.

    Returns:
    - img_bg: (flattened) output image, usually its background.
    - i: current iteration (mod. 100) when the convergence was reached.
    |----------------------------
    """
    start = time.time()
    L = np.sqrt(8)
    sigma = 1 / tau / L**2

    M, N = image.shape
    image = image.flatten()
    # primal variable
    img_bg = np.copy(image)
    # dual variable
    p = np.zeros(M * N * 2)

    if nabla_comp == "both":
        nabla = make_nabla(M, N)
    elif nabla_comp == "x":
        nabla = make_nabla_x(M, N)
    elif nabla_comp == "y":
        nabla = make_nabla_y(M, N)
    else:
        print("Error, select one of the three options.")

    for i in range(iterations + 1):
        img_old = np.copy(img_bg)
        img_bg = img_bg - tau * nabla.T @ p

        if algo == 1:  # l1-TV
            img_bg = image + np.maximum(0.0, np.abs(img_bg-image) - tau*lam) * np.sign(img_bg-image)
            alpha = 0
            # overrelaxation
            if overrelax == True:
                img_old = 2 * img_bg - img_old
        if algo == 2:  # Huber-ROF
            img_bg = (img_bg + tau*lam*image)/(1 + tau*lam)
            # overrelaxation
            if overrelax == True:
                img_old = 2 * img_bg - img_old

        p = (p + sigma*(nabla@img_old))/(1 + sigma*alpha)

        # proximal map
        p = p.reshape(2, M * N)
        norm_p = np.sqrt(p[0, :] ** 2 + p[1, :] ** 2)
        denom = np.maximum(1, norm_p)
        p = p / denom[np.newaxis, :]
        p = p.flatten()

        if np.mod(i, 1000) == 0:
            print("iter:", i)
            if gif == True:
                result = img_bg.reshape(M, N)
                result = cv2.imwrite(
                    r"results/last_gif/" + str(i) + ".png", result * 255
                )  # input imgs norm. to 1!

        if i > 0:
            if np.mod(i, 100) == 0:
                rmse = np.sqrt(np.mean((img_bg - img_old) ** 2))
                print("Iteration: {},  RMSE: {}".format(i, rmse))
                if rmse < err and err != 0:
                    print("Total # iterations: {}".format(i))
                    break

    end = time.time()
    print("Execution time:", round(end - start, 1), "s.")
    return img_bg, i


def l1TGV(image, iterations, nabla_comp, lam, tau, alpha1, alpha2, err, gif=False):
    """
    Applies l1-TGV to an image.
    |----------------------------
    Parameters:
    - image - input image. Grayscale at least for now.
    - iterations - number of iterations.
    - nabla_comp - choose either 'both', 'x', 'y' to select the nonzero components
                   of the operator.
    - lam - parameter lambda in TGV.
    - tau -    //     tau      // .
    - alpha1 & alpha2 - weight of 1st order & 2nd order terms, respectively.
    - err: RMSE threshold for convergence. Set to 0 to turn execute a 'iteration' number of iterations.
    - gif - boolean to choose whether to save or not the imgs for the future gif.

    Output:
    - img_bg: (flattened) output image, usually its background.
    |----------------------------
    """
    start = time.time()
    M, N = image.shape
    f = np.copy(image.flatten())
    L = np.sqrt(12)

    sigma = 1 / tau / L**2

    # clean image (primal variable)
    u = np.zeros(M * N * 3)
    u[: M * N] = f

    # dual variable
    p = np.zeros(M * N * 6)

    # make nabla operator
    if nabla_comp == "both":
        nabla = make_nabla(M, N)
    elif nabla_comp == "x":
        nabla = make_nabla_x(M, N)
    elif nabla_comp == "y":
        nabla = make_nabla_y(M, N)
    else:
        print("Error, select one of the three options.")

    Z = scipy.sparse.coo_matrix((2 * M * N, M * N))
    I = scipy.sparse.eye(2 * M * N)

    K1 = scipy.sparse.hstack([nabla, -I])
    K2 = scipy.sparse.hstack([Z, nabla, Z])
    K3 = scipy.sparse.hstack([Z, Z, nabla])

    K = scipy.sparse.vstack([K1, K2, K3])

    for i in range(iterations + 1):
        # primal update
        u_ = np.copy(u)
        u = u - tau * (K.T @ p)

        # proximal maps
        u[: M * N] = f + np.maximum(0.0, np.abs(u[: M * N] - f) - tau * lam) * np.sign(
            u[: M * N] - f
        )

        # overrelaxation
        u_ = 2 * u - u_

        # dual update
        p = p + sigma * (K @ u_)

        # proximal maps
        p1 = p[: 2 * M * N].reshape(2, M * N)
        norm_p = np.sqrt(p1[0, :] ** 2 + p1[1, :] ** 2)
        denom = np.maximum(1, norm_p / alpha1)
        p1 = p1 / denom[np.newaxis, :]
        p[: 2 * M * N] = p1.flatten()

        p2 = p[2 * M * N :].reshape(4, M * N)
        norm_p = np.sqrt(p2[0, :] ** 2 + p2[1, :] ** 2 + p2[2, :] ** 2 + p2[3, :] ** 2)
        denom = np.maximum(1, norm_p / alpha2)
        p2 = p2 / denom[np.newaxis, :]
        p[2 * M * N :] = p2.flatten()

        if np.mod(i, 1000) == 0:
            print("TGV: iter = ", i)
            if gif == True:
                result = u[: M * N].reshape(M, N)
                result = cv2.imwrite(
                    r"test_results/last_gif/" + str(i) + ".png", result * 255
                )  # input imgs norm. to 1!

        if i > 0:
            if np.mod(i, 100) == 0:
                rmse = np.sqrt(np.mean((u - u_) ** 2))
                print("Iteration: {},  RMSE: {}".format(i, rmse))
                if rmse < err and err != 0:
                    print("Total # iterations: {}".format(i))
                    break

    img_bg = np.copy(u)
    end = time.time()
    print("Execution time:", round(end - start, 1), "s.")
    return img_bg, i