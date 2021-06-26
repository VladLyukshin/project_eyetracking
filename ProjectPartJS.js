export function video_player() {
    // функция, отвечающая за запуск видео
    var video = document.getElementById('video');
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true }).then(function (stream) {
            video.srcObject = stream;
            video.play();
        });
    }
}

export function include_model() {
    // подключение модели, и затем запуск функции, отвечающей за отрисовку канваса, и обработки фото с отправкой метрики на бэк
    Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('static/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('static/models'),
    ]).then(take_photo_and_request());
}

async function Get_Metric(model, displaySize) {
    tf.engine().startScope();
    // проверка модели Tiny
    let detector = new faceapi.TinyFaceDetectorOptions();
    const detections = await faceapi.detectAllFaces(video, detector).withFaceLandmarks();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    detector = null;

    if (resizedDetections['0']) {
        const positions = resizedDetections['0']['landmarks']['positions'];
        // Угол поворота изображения
        let angle = Math.acos(
            Math.abs(positions[6 + 36]['_x'] - positions[3 + 36]['_x']) /
                Math.sqrt(
                    (positions[6 + 36]['_x'] - positions[3 + 36]['_x']) *
                        (positions[6 + 36]['_x'] - positions[3 + 36]['_x']) +
                        (positions[6 + 36]['_y'] - positions[3 + 36]['_y']) *
                            (positions[6 + 36]['_y'] - positions[3 + 36]['_y']),
                ),
        );
        if (positions[3 + 36]['_y'] > positions[6 + 36]['_y']) angle = -angle;
        const dist_left = Math.sqrt(
            Math.pow(positions[36]['_x'] - positions[0]['_x'], 2) +
                Math.pow(positions[36]['_y'] - positions[0]['_y'], 2),
        );
        const dist_right = Math.sqrt(
            Math.pow(positions[16]['_x'] - positions[45]['_x'], 2) +
                Math.pow(positions[16]['_y'] - positions[45]['_y'], 2),
        );
        const example = tf.browser.fromPixels(video);
        // Тензор левого глаза
        const left_eye = example.slice(
            [
                Math.round(
                    positions[1 + 36]['_y'] -
                        (2 / 3) * (positions[5 + 36]['_y'] - positions[1 + 36]['_y']),
                ),
                Math.round(
                    positions[1 + 36]['_x'] -
                        (2 / 3) * (positions[3 + 36]['_x'] - positions[0 + 36]['_x']),
                ),
                0,
            ],
            [
                Math.round(
                    positions[4 + 36]['_y'] +
                        (4 / 3) * positions[5 + 36]['_y'] -
                        (7 / 3) * positions[1 + 36]['_y'],
                ),
                Math.round(
                    positions[4 + 36]['_x'] +
                        (4 / 3) * positions[3 + 36]['_x'] -
                        positions[1 + 36]['_x'] -
                        (4 / 3) * positions[0 + 36]['_x'],
                ),
                3,
            ],
        );
        // Тензор правого глаза
        const right_eye = example.slice(
            [
                Math.round(
                    positions[7 + 36]['_y'] -
                        (2 / 3) * (positions[5 + 36]['_y'] - positions[1 + 36]['_y']),
                ),
                Math.round(
                    positions[7 + 36]['_x'] -
                        (2 / 3) * (positions[3 + 36]['_x'] - positions[0 + 36]['_x']),
                ),
                0,
            ],
            [
                Math.round(
                    positions[4 + 36]['_y'] +
                        (4 / 3) * positions[5 + 36]['_y'] -
                        (7 / 3) * positions[1 + 36]['_y'],
                ),
                Math.round(
                    positions[4 + 36]['_x'] +
                        (4 / 3) * positions[3 + 36]['_x'] -
                        positions[1 + 36]['_x'] -
                        (4 / 3) * positions[0 + 36]['_x'],
                ),
                3,
            ],
        );

        // Конкатенация
        let concat_eyes = tf.concat([left_eye, right_eye], 1);
        // Загрузка модели
        let tensor_1352 = tf.image
            .resizeBilinear(concat_eyes, [30, 15])
            .reshape([1350])
            .concat(tf.tensor1d([dist_left, dist_right]));
        let min_t = tensor_1352.min();
        let max_t = tensor_1352.max();
        let normilized = tensor_1352.sub(min_t).div(max_t);
        let prediction = model.predict(normilized.expandDims(1).expandDims(0));
        let result = prediction.dataSync()[0] * 100;
        if (Math.abs(dist_right - dist_left) > 40 && result < 90) {
            result += 10;
        }

        tf.engine().endScope();
        return result;
    } else {
        tf.engine().endScope();
        return 100;
    }
}

async function stream() {
    var canvas = document.getElementById('canvas');
    var context = canvas.getContext('2d');
    var timer;
    const model = await tf.loadLayersModel('static/tfjsmodel/model.json');
    const displaySize = { width: canvas.width, height: canvas.height };
    document.getElementById('stop_stream').addEventListener('click', function () {
        clearTimeout(timer);
    });
    async function cam_interval() {
        clearTimeout(timer);
        timer = setTimeout(async function () {
            context.drawImage(video, 0, 0, 640, 480);
            Get_Metric(model, displaySize).then(function (result) {
                if (result < 15) {
                    document.getElementById('result').innerHTML =
                        String(result) + String(', смотрит!');
                } else {
                    document.getElementById('result').innerHTML =
                        String(result) + String(', НЕ смотрит!');
                }
                console.log(result);
                var url_canvas = canvas.toDataURL();
                $.ajax({
                    type: 'POST',
                    url: '/',
                    data: {
                        name: String(document.getElementById('name').value) + ',' + String(result),
                        picture: url_canvas,
                    },
                    success: function (msg) {
                        console.log(msg);
                    },
                    error: function (msg) {
                        console.log('Ошибка!');
                    },
                });
                url_canvas = null;
            });
            clearTimeout(timer);
            cam_interval();
        }, 5000);
    }
    cam_interval();
}

export async function take_photo_and_request() {
    // функция, отвечающая за канвас и отправку метрики на бэк
    document.getElementById('translation-of-photo').addEventListener(
        'click',
        function () {
            document.getElementById('continue_stream').addEventListener('click', stream);
        },
        { once: true },
    );
}

export function define_device() {
    // определение устройства, которое используют
    if (
        /Android|webOS|iPhone|iPad|iPod|BlackBerry|BB|PlayBook|IEMobile|Windows Phone|Kindle|Silk|Opera Mini/i.test(
            navigator.userAgent,
        )
    ) {
        console.log('Пользователь использует мобильное устройство (телефон или планшет).');
    } else {
        console.log('Пользователь использует ПК.');
    }
}




