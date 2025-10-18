from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'


# ---------- Helper Functions ----------
def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def log_transform(img):
    imgf = img.astype(np.float32)
    out = np.log1p(imgf)
    out = out / np.max(out) * 255.0
    return np.uint8(np.clip(out, 0, 255))


def power_law_transform(img, gamma=0.6):
    imgf = img.astype(np.float32) / 255.0
    out = np.power(imgf, gamma)
    out = out * 255.0
    return np.uint8(np.clip(out, 0, 255))


def make_psf(size=21, sigma=5):
    k = cv2.getGaussianKernel(size, sigma)
    psf = k @ k.T
    return psf


def pad_psf(psf, shape):
    ph, pw = psf.shape
    h, w = shape
    pad = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2 - ph // 2, w // 2 - pw // 2
    pad[cy:cy+ph, cx:cx+pw] = psf
    return pad


def inverse_filter(img, psf, eps=1e-3):
    G = np.fft.fft2(img.astype(np.float32))
    H = np.fft.fft2(psf)
    H_mag = np.abs(H)
    H[H_mag < eps] = eps
    F_est = G / H
    f_est = np.fft.ifft2(F_est)
    return np.uint8(np.clip(np.real(f_est), 0, 255))


def pseudo_inverse(img, psf, K=0.01):
    G = np.fft.fft2(img.astype(np.float32))
    H = np.fft.fft2(psf)
    H_conj = np.conjugate(H)
    denom = (np.abs(H)**2) + K
    F_est = (H_conj * G) / denom
    f_est = np.fft.ifft2(F_est)
    return np.uint8(np.clip(np.real(f_est), 0, 255))


def wiener_filter(img, psf, K=0.01):
    G = np.fft.fft2(img.astype(np.float32))
    H = np.fft.fft2(psf)
    H_conj = np.conjugate(H)
    denom = (np.abs(H)**2) + K
    F_est = (H_conj / denom) * G
    f_est = np.fft.ifft2(F_est)
    return np.uint8(np.clip(np.real(f_est), 0, 255))


def btc_compress(img, block_size=4):
    """Block Truncation Coding (improved visible version)"""
    gray = to_gray(img).astype(np.float32)
    h, w = gray.shape
    out = np.zeros_like(gray)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            if block.size == 0:
                continue
            m = np.mean(block)
            var = np.var(block)
            if var == 0:
                out[i:i+block_size, j:j+block_size] = m
                continue
            q = block >= m
            n = np.count_nonzero(q)
            if n == 0 or n == block.size:
                out[i:i+block_size, j:j+block_size] = m
                continue
            sigma = np.sqrt(var)
            high = m + sigma * np.sqrt(n / (block.size - n))
            low = m - sigma * np.sqrt((block.size - n) / n)
            rec = np.where(q, high, low)
            out[i:i+block_size, j:j+block_size] = rec
    return np.uint8(np.clip(out, 0, 255))


# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    file = request.files.get('file')
    operation = request.form.get('operation')

    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')

    if file and file.filename != '':
        file.save(upload_path)
    elif not os.path.exists(upload_path):
        return render_template('index.html', output_image=None)

    img = cv2.imread(upload_path)
    processed = img.copy()
    op_name = "Original"

    # ========== Processing ==========
    if operation == 'gray':
        processed = to_gray(img)
        op_name = "Grayscale"

    elif operation == 'blur':
        processed = cv2.GaussianBlur(img, (15, 15), 0)
        op_name = "Gaussian Blur"

    elif operation == 'edge':
        processed = cv2.Canny(to_gray(img), 100, 200)
        op_name = "Edge Detection"

    elif operation == 'rotate':
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 45, 1)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        processed = cv2.warpAffine(img, M, (new_w, new_h))
        op_name = "Rotation (45°)"

    elif operation == 'translate':
        h, w = img.shape[:2]
        M = np.float32([[1, 0, 100], [0, 1, 50]])
        processed = cv2.warpAffine(img, M, (w + 100, h + 50))
        op_name = "Translation"

    elif operation == 'scale':
        processed = cv2.resize(img, None, fx=1.5, fy=1.5)
        op_name = "Scaling (1.5x)"

    elif operation == 'brightness':
        processed = cv2.convertScaleAbs(img, alpha=1, beta=60)
        op_name = "Brightness Increased"

    elif operation == 'contrast':
        processed = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        op_name = "Contrast Enhanced"

    elif operation == 'gray_slice_preserve':
        gray = to_gray(img)
        low, high = 100, 200
        mask = (gray >= low) & (gray <= high)
        out = np.copy(gray)
        out[mask] = 255
        processed = out
        op_name = "Gray-Level Slicing (Preserve)"

    elif operation == 'gray_slice_nopreserve':
        gray = to_gray(img)
        low, high = 100, 200
        out = np.zeros_like(gray)
        mask = (gray >= low) & (gray <= high)
        out[mask] = 255
        processed = out
        op_name = "Gray-Level Slicing (No Background)"

    elif operation == 'log':
        processed = log_transform(to_gray(img))
        op_name = "Logarithmic Transform"

    elif operation == 'power':
        processed = power_law_transform(to_gray(img), 0.6)
        op_name = "Power-Law (Gamma) Transform"

    elif operation == 'zoom':
        h, w = img.shape[:2]
        crop = img[h//4:h*3//4, w//4:w*3//4]
        processed = cv2.resize(crop, (w, h))
        op_name = "Zoom (Center Crop)"

    elif operation == 'inverse_filter':
        gray = to_gray(img)
        h, w = gray.shape
        psf = pad_psf(make_psf(15, 5), (h, w))
        processed = inverse_filter(gray, psf)
        op_name = "Inverse Filtering"

    elif operation == 'pseudo_inverse':
        gray = to_gray(img)
        h, w = gray.shape
        psf = pad_psf(make_psf(15, 5), (h, w))
        processed = pseudo_inverse(gray, psf)
        op_name = "Pseudo-Inverse Filtering"

    elif operation == 'wiener':
        gray = to_gray(img)
        h, w = gray.shape
        psf = pad_psf(make_psf(15, 5), (h, w))
        processed = wiener_filter(gray, psf)
        op_name = "Wiener Filtering"

    elif operation == 'median':
        processed = cv2.medianBlur(img, 5)
        op_name = "Median Filtering"

    elif operation == 'laplacian':
        gray = to_gray(img)
        lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        processed = cv2.convertScaleAbs(lap)
        op_name = "Laplacian Operator"

    elif operation == 'dog':
        gray = to_gray(img)
        g1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
        g2 = cv2.GaussianBlur(gray, (9, 9), 2.0)
        dog = cv2.subtract(g1, g2)
        dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        processed = np.uint8(dog_norm)
        op_name = "Difference of Gaussian (DoG)"

    elif operation == 'btc':
        processed = btc_compress(img)
        op_name = "BTC Compression"

    elif operation == 'histogram':
        gray = to_gray(img)
        processed = cv2.equalizeHist(gray)
        op_name = "Histogram Equalization"

    else:
        processed = img
        op_name = "Original Image"

    cv2.imwrite(output_path, processed)

    return render_template('index.html',
                           uploaded_image='uploaded.jpg',
                           output_image='output.jpg',
                           operation_name=op_name)


# ---------- Run Server ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
