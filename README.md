# laba4
Формируется матрица F следующим образом: скопировать в нее А и если А симметрична относительно главной диагонали, 
то поменять местами С и  В симметрично, иначе B и Е поменять местами несимметрично. При этом матрица А не меняется. 
После чего если определитель матрицы А больше суммы диагональных элементов матрицы F, то вычисляется выражение: A-1*AT – K * F-1, 
иначе вычисляется выражение (AТ +G-FТ)*K, где G-нижняя треугольная матрица, полученная из А. 
Выводятся по мере формирования А, F и все матричные операции последовательно.
