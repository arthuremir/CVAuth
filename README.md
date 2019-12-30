Версия написана с использованием tkinter.
Есть возможность выбора включения/отключения модулей, а также возможность регистрации пользователя (занесение его имени в базу и сохранение лица с последующим пересчетом эмбеддинга пользователя)


Процесс авторизации
![Face verification](gifs/almost_final.gif)


Отчет (предварительный):

Пользователь вводит свой логин, в случае наличия такового в базе включается камера. В случае успешного распознавания лица пользователя ему предлагается показать жест в специальную область.

Использованные алгоритмы:
- распознавание лиц: MTCNN+ArcFace
- анализ позы: ResNet-50_FPN_3x из detectron2
- разспознавание жестов: ~~детектор, обученный на EgoGestures~~ 
                         сегментация вычитанием фона и (предварительно) классификация с помощью resnet18, обученной на [HGRD](https://www.kaggle.com/gti-upm/leapgestrecog) и собственном датасете. 

Работоспособность проверялась в боевых условиях (веб-камерой).


Задачи:
- !!! обучить классификатор жестов по сегментированному бинарному изображению
- увеличить fps
- (DONE) файл с метками обновления эмбеддингов
- (DONE) добавить логику авторизации
- (DONE) переписать логику tkinter в объектно-ориентированном стиле
- (DONE) доразобраться с гитом и правильно делать коммиты






Old version:

Face verification  
![Face verification](gifs/face.gif)



Pose estimation  
![Pose estimation](gifs/pose.gif)



Hand localization  
![Hand localization](gifs/hand.gif)



Sighning up  
![Sighning up](gifs/new_user_registration.gif)

Based on:
- https://github.com/facebookresearch/detectron2
- https://github.com/TreB1eN/InsightFace_Pytorch
- https://github.com/zllrunning/hand-detection.PyTorch
