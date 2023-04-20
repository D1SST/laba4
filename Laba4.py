"""
Формируется матрица F следующим образом: скопировать в нее А и если А симметрична относительно главной диагонали,
то поменять местами С и  В симметрично, иначе B и Е поменять местами несимметрично. При этом матрица А не меняется.
После чего если определитель матрицы А больше суммы диагональных элементов матрицы F,
то вычисляется выражение: A-1*AT – K * F-1,иначе вычисляется выражение (AТ +G-FТ)*K, где G-нижняя треугольная матрица,
полученная из А. Выводятся по мере формирования А, F и все матричные операции последовательно.
"""

import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

try:
    k = int(input("Введите число K, являющееся коэффициентом при умножении: "))
    n = int(input("Введите число число N, большее 3, являющееся порядком квадратной матрицы: "))
    while n <= 3:
        n = int(input("Вы ввели число, неподходящее по условию, введите число N, большее или равное 3:\n"))

    np.set_printoptions(linewidth=1000)

    A = np.random.randint(-10.0, 10.0, (n, n))

    print("\nМатрица A:\n", A)

    # Создание подматриц
    submatrix_length = n//2
    sub_matrix_B = np.array(A[:submatrix_length, :submatrix_length])
    sub_matrix_C = np.array(A[:submatrix_length, submatrix_length+n % 2:n])
    sub_matrix_E = np.array(A[submatrix_length+n % 2:n, submatrix_length+n % 2:n])

    # Создание матрицы F
    F = A.copy()
    print("\nМатрица F:\n", F)


    def isSymmetric(mat, N):
        for i in range(N):
            for j in range(N):
                if mat[i][j] != mat[j][i]:
                    return False
        return True
    print("Проверка матрицы А на симметричность... ")
    if (isSymmetric(A, n)):
        print("Результат: симметрична. \nМеняем симметрично B и C.")
        F[:submatrix_length, submatrix_length + n % 2:n] = sub_matrix_B[:submatrix_length, ::-1]
        F[:submatrix_length, :submatrix_length] = sub_matrix_C[:submatrix_length, ::-1]
    else:
        print("Результат: несимметрична. \nМеняем несимметрично B и E.")
        F[:submatrix_length, :submatrix_length] = sub_matrix_E
        F[submatrix_length + n % 2:n, submatrix_length + n % 2:n] = sub_matrix_B

    print("\nОтформатированная матрица F:\n", F)
    # Вычисляем выражение
    try:
        if np.linalg.det(A) > sum(np.diagonal(F)):
            print("\nРезультат выражения A^-1 * A^T – K * F^-1:\n", np.linalg.inv(A)*A.transpose() - np.linalg.inv(F)*k)
        else:
            G = np.tri(n)*A
            print("\nРезультат выражения (A^Т + G - F^Т) * K:\n", (A.transpose() + G - F.transpose()) * k)

    except np.linalg.LinAlgError:
        print("Одна из матриц является вырожденной (определитель равен 0), поэтому обратную матрицу найти невозможно.")

    print("\nМатрица, которая используется при построение графиков:\n", F)
    # Использование библиотеки matplotlib
    av = [np.mean(abs(F[i, ::])) for i in range(n)]
    av = int(sum(av))
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    x = list(range(1, n+1))
    for j in range(n):
        y = list(F[j, ::])
        axs[0, 0].plot(x, y, ',-', label=f"{j+1} строка.")
        axs[0, 0].set(title="График с использованием функции plot:", xlabel='Номер элемента в строке', ylabel='Значение элемента')
        axs[0, 0].grid()
        axs[0, 1].bar(x, y, 0.4, label=f"{j+1} строка.")
        axs[0, 1].set(title="График с использованием функции bar:", xlabel='Номер элемента в строке', ylabel='Значение элемента')
        if n <= 10:
            axs[0, 1].legend(loc='lower right')
            axs[0, 1].legend(loc='lower right')
    explode = [0]*(n-1)
    explode.append(0.1)
    sizes = [round(np.mean(abs(F[i, ::])) * 100/av, 1) for i in range(n)]
    axs[1, 0].set_title("График с ипользованием функции pie:")
    axs[1, 0].pie(sizes, labels=list(range(1, n+1)), explode=explode, autopct='%1.1f%%', shadow=True)
    def heatmap(data, row_labels, col_labels, ax, cbar_kw={}, **kwargs):
        im = ax.imshow(data, **kwargs)
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
        return im, cbar
    def annotate_heatmap(im, data = None, textcolors=("black","white"), threshold=0):
        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()
        kw = dict(horizontalalignment="center", verticalalignment="center")
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(data[i, j] > threshold)])
                text = im.axes.text(j, i, data[i, j], **kw)
                texts.append(text)
        return texts
    im, cbar = heatmap(F, list(range(n)), list(range(n)), ax=axs[1, 1], cmap="magma_r")
    texts = annotate_heatmap(im)
    axs[1, 1].set(title="Создание аннотированных тепловых карт:", xlabel="Номер столбца", ylabel="Номер строки")
    plt.suptitle("Использование библиотеки matplotlib")
    plt.tight_layout()
    plt.show()
    # использование библиотеки seaborn
    number_row = []
    for i in range(1, n+1):
        number_row += [i]*n
    number_item = list(range(1, n+1))*n
    df = pd.DataFrame({"Значения": F.flatten(), "Номер строки": number_row, "Номер элемента в строке": number_item})
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    plt.subplot(221)
    plt.title("Использование функции lineplot")
    sns.lineplot(x="Номер элемента в строке", y="Значения", hue="Номер строки", data=df, palette="Set2")
    plt.subplot(222)
    plt.title("Использование функции boxplot")
    sns.boxplot(x="Номер строки", y="Значения", palette="Set2", data=df)
    plt.subplot(223)
    plt.title("Использование функции kdeplot")
    sns.kdeplot(data=df, x="Номер элемента в строке", y="Значения", hue="Номер строки", palette="Set2")
    plt.subplot(224)
    plt.title("Использование функции heatmap")
    sns.heatmap(data=F, annot=True, fmt="d", linewidths=.5)
    plt.suptitle("Использование библиотеки seaborn")
    plt.tight_layout()
    plt.show()

except ValueError:
    print("\nВведенный символ не является числом. Перезапустите программу и введите число.")