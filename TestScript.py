import mnist_loader
import NnAlgorithm


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = NnAlgorithm.Network([784, 30, 30, 10])
net.gradient_descent(training_data, 10, 10, 3.0, test_data=test_data)

# n = 0

# vals = []
# dats = []
# for val, dat in validation_data:
#     vals.append(val)
#     dats.append(dat)
# while n == 0:
#     entrada = input('Ingrese un valor entre 0 a ' + str(len(vals)) + ' para realizar la prueba: ')
#     entrada = int(entrada)
#     # Se grafica la seleccion
#
#     dato = vals[entrada]
#     dato_array = np.array(dato)
#     dato_array = dato_array.reshape(28, 28)
#     plt.imshow(dato_array, plt.get_cmap('gray'))
#     plt.show()
#
#     res = net.feedforward(vals[entrada])
#     print(np.round(res))
#     print(dats[entrada])
#     salir = input('Desea salir del ciclo? y / n: ')
#     if salir == 'y':
#         n = 1
#     else:
#         n = 0
