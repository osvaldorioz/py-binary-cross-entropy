#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Función para calcular la pérdida de Binary Cross-Entropy
torch::Tensor binary_cross_entropy(torch::Tensor input, torch::Tensor target) {
    // Aplicar la función sigmoide al input
    auto sigmoid_input = torch::sigmoid(input);
    // Calcular la pérdida de Binary Cross-Entropy
    auto loss = torch::binary_cross_entropy(sigmoid_input, target);
    return loss;
}

// Enlace de la función con Pybind11
PYBIND11_MODULE(bce_module, m) {
    m.def("binary_cross_entropy", &binary_cross_entropy, "Calcula la pérdida de Binary Cross-Entropy",
          py::arg("input"), py::arg("target"));
}
