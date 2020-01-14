Отчет

Версия написана с использованием tkinter. <br>
Есть возможность включения/отключения расчета позы, а также возможность регистрации пользователя
(занесение его имени и жеста в базу и сохранение лица с последующим пересчетом эмбеддинга пользователя)

Пользователь вводит свой логин, в случае наличия такового в базе включается камера. <br>
В случае успешного распознавания лица пользователя ему предлагается показать жест в специальную область.

Использованные алгоритмы:
- распознавание лиц: MTCNN+ArcFace
- анализ позы: ResNet-50_FPN_3x из detectron2
- разспознавание жестов: сегментация вычитанием фона и классификация с помощью ResNet-18, предобученной на ImageNet и обученной на собственном датасете. <br>
<br>

Датасет бинаризованных изображений семи жестов (+ шум) был собран с веб-камеры.
Состоит из ~900 изображений:<br>

Один палец - 98<br>
Два пальца - 97<br>
Три пальца - 225<br>
Четыре пальца - 103<br>
Пять пальцев - 99<br>
Жест "Ок" - 102<br>
Кулак - 100<br>
Шум (нет жеста) - 102<br>


Аугментации:<br>
<code>transforms.Compose([<br>
&nbsp;&nbsp;&nbsp;&nbsp;transforms.RandomAffine(25,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(0.15, 0.15),<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(0.7, 1.1)),<br>
&nbsp;&nbsp;&nbsp;&nbsp;transforms.RandomHorizontalFlip(),<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;transforms.ToTensor()<br>
])
</code>
<br><br>

Точность классификации жестов порядка 95% (ошибки в основном в неочевидных случаях в классе "шум")

Работоспособность проверялась в боевых условиях (веб-камерой).

Использованные библиотеки:
- https://github.com/facebookresearch/detectron2
- https://github.com/TreB1eN/InsightFace_Pytorch
- https://github.com/zllrunning/hand-detection.PyTorch
