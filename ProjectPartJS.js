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

async function Get_Metric(blazeface_detector, model) {
    tf.engine().startScope();

    // Детектирование лица
    const predictions = await blazeface_detector.estimateFaces(video, false);
    // Получаем первоначальное изображение в виде тензора
    let example = tf.browser.fromPixels(video);

    if (predictions.length > 0) {
        // Считаем расстояния:
        // Расстояние d - между точками глаз
        let d = Math.sqrt(
            Math.pow(predictions[0].landmarks[1][0] - predictions[0].landmarks[0][0], 2) +
                Math.pow(predictions[0].landmarks[1][1] - predictions[0].landmarks[0][1], 2),
        );

        // Расстояние d1 - между точками правого глаза и рта
        let d1 = Math.sqrt(
            Math.pow(predictions[0].landmarks[3][0] - predictions[0].landmarks[0][0], 2) +
                Math.pow(predictions[0].landmarks[3][1] - predictions[0].landmarks[0][1], 2),
        );

        // Расстояние dist1 - между точками правого глаза и правого уха
        let dist1 = Math.sqrt(
            Math.pow(predictions[0].landmarks[5][0] - predictions[0].landmarks[1][0], 2) +
                Math.pow(predictions[0].landmarks[5][1] - predictions[0].landmarks[1][1], 2),
        );

        // Расстояние dist2 - между точками левого глаза и левого уха
        let dist2 = Math.sqrt(
            Math.pow(predictions[0].landmarks[4][0] - predictions[0].landmarks[0][0], 2) +
                Math.pow(predictions[0].landmarks[4][1] - predictions[0].landmarks[0][1], 2),
        );

        // Угол поворота
        var angle = Math.atan(
            (predictions[0].landmarks[1][1] - predictions[0].landmarks[0][1]) /
                (predictions[0].landmarks[1][0] - predictions[0].landmarks[0][0]),
        );

        // Осуществляем поворот
        const rotated = tf.image
            .rotateWithOffset(example.toFloat().expandDims(0), angle, 0)
            .squeeze()
            .toInt();

        // Координаты правого глаза
        let x0 = predictions[0].landmarks[0][0];
        let y0 = predictions[0].landmarks[0][1];

        // Координаты левого глаза
        let x1 = predictions[0].landmarks[1][0];
        let y1 = predictions[0].landmarks[1][1];

        // Матрица поворота
        let M = [
            [Math.cos(angle), Math.sin(angle)],
            [-Math.sin(angle), Math.cos(angle)],
        ];

        // Контроль выделения правого глаза
        let y_start = M[1][0] * (x0 - 320) + M[1][1] * (y0 - 240) + 240 - (10 / 24) * d1;
        let x_start = M[0][0] * (x0 - 320) + M[0][1] * (y0 - 240) + 320 - (5 / 15) * d1;
        if (y_start < 0) y_start = 0;
        if (x_start < 0) x_start = 0;
        let y_end = y_start + (14 / 24) * d1;
        let x_end = x_start + (10 / 15) * d1;
        if (y_end > video.height) y_end = video.height;
        if (x_end > video.width) x_end = video.width;

        // Тензор правого глаза
        let right_eye = rotated.slice(
            [Math.round(y_start), Math.round(x_start), 0],
            [Math.round(y_end - y_start), Math.round(x_end - x_start), 3],
        );

        // Контроль выделения левого глаза
        y_start = M[1][0] * (x1 - 320) + M[1][1] * (y1 - 240) + 240 - (10 / 24) * d1;
        x_start = M[0][0] * (x1 - 320) + M[0][1] * (y1 - 240) + 320 - (5 / 15) * d1;
        if (y_start < 0) y_start = 0;
        if (x_start < 0) x_start = 0;
        y_end = y_start + (14 / 24) * d1;
        x_end = x_start + (10 / 15) * d1;
        if (y_end > video.height) y_end = video.height;
        if (x_end > video.width) x_end = video.width;

        // Тензор левого глаза
        let left_eye = rotated.slice(
            [Math.round(y_start), Math.round(x_start), 0],
            [Math.round(y_end - y_start), Math.round(x_end - x_start), 3],
        );

        // Конкатенация областей глаз
        let concat_eyes = tf.concat([right_eye, left_eye], 1);

        // Отправка в нейросеть
        const tensor_1354 = tf.image
            .resizeBilinear(concat_eyes, [30, 15])
            .reshape([1350])
            .concat(tf.tensor1d([dist1, dist2, d, Math.abs(d - d1)]));
        const min_t = tensor_1354.min();
        const max_t = tensor_1354.max();
        const normilized = tensor_1354.sub(min_t).div(max_t.sub(min_t));
        const prediction = model.predict(normilized.expandDims(1).expandDims(0));
        let result = prediction.dataSync()[0] * 100;
        tf.engine().endScope();
        return result;
    } else {
        tf.engine().endScope();
        return 0;
    }
}

async function stream() {
    var canvas = document.getElementById('canvas');
    var context = canvas.getContext('2d');
    var timer;
    const blazeface_detector = await blazeface.load();
    const model = await tf.loadLayersModel('static/tfjsmodel/model.json');
    document.getElementById('stop_stream').addEventListener('click', function () {
        clearTimeout(timer);
    });
    async function cam_interval() {
        clearTimeout(timer);
        timer = setTimeout(async function () {
            context.drawImage(video, 0, 0, 640, 480);
            Get_Metric(blazeface_detector, model).then(function (result) {
                if (result >= 70) {
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
        }, 500);
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




