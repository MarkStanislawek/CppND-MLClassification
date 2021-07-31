#include "Graphics.h"
#include "ClassificationDemo.h"
#include <iostream>

int main() {
    ClassificationDemo *clsDemo = new ClassificationDemo();
    clsDemo->RunDemo();

    Graphics *graphics = new Graphics();
    graphics->SetDemonstrator(clsDemo);
    graphics->Start();

    std::cout << "ML Image Classification Demo.  Images and their classification will rotate on display."  << std::endl;
    std::cout << "Press any key to exit gracefully." << std::endl;
    std::cin.get();

    graphics->Stop();
    clsDemo->Stop();
}