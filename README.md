# vs-kangaroo
Pollard's kangaroo algoritm tool. Use work files to save the distance traveled.
Added function for dividing public key.
# Puzzle 32 BTC
Public key bit 130 \
03633cbe3ec02b9401c5effa144c5b4d22f87940259634858fc7e59b1c09937852 
## Run cmd:
$ unzip vs-kangaroo.zip \
$ cd vs-kangaroo \
$ make gpu=1 ccap=75 all \
$ chmod 755 vs-kangaroo-multi-divide \
$ ./vs-kangaroo-multi-divide -drive -work -v 2 -p 40 -gpu -dp 20 -bits 200000000000000000000000000000000:3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF 03633cbe3ec02b9401c5effa144c5b4d22f87940259634858fc7e59b1c09937852 

## Check low bits
$ ./vs-kangaroo-multi-divide -v 2 -gpu -dp 14 -bits 65 0230210c23b1a047bc9bdbb13448e67deddc108946de6de639bcc75d47c0216b1b \
$ ./vs-kangaroo-multi-divide -v 2 -gpu -dp 14 -bits 75 03726b574f193e374686d8e12bc6e4142adeb06770e0a2856f5e4ad89f66044755 

----------------------------------------------------------------------------------------------------------------
# Compilation

## Windows

Intall CUDA SDK and build OpenSSL, open VanitySearch.sln in Visual C++ 2017. \
You may need to reset your *Windows SDK version* in project properties. \
In Build->Configuration Manager, select the *Release* configuration. \
Build OpenSSL: \
Install Perl x64 for MS Windows from https://strawberryperl.com \
Link: https://strawberryperl.com/download/5.32.1.1/strawberry-perl-5.32.1.1-64bit.msi \
Install Netwide Assembler (NASM).  \
Download NASM x64 https://www.nasm.us/pub/nasm/releasebuilds/2.16.01/win64/nasm-2.16.01-installer-x64.exe \
Add Path C:\Program Files\NASM\; \
Add Path C:\Strawberry\perl\bin\; \
Add PATHEXT .PL; before .BAT; \
those. - .COM;.EXE;.PL;.BAT; \
And be sure to restart your PC. \
Download the library from the official website openssl-1.0.1a.tar.gz \
http://www.openssl.org/source/old/1.0.1/openssl-1.0.1a.tar.gz \
Unpack openssl-1.0.1a.tar.gz into the openssl-1.0.1a directory and copy its contents to the directory: \
c:\openssl-src-64 \
Run Command Prompt Visual Studio - x64 Native Tools Command Prompt for VS 2017 as administrator \
Run commands: \
cd C:\openssl-src-64 \
perl Configure VC-WIN64A --prefix=C:\Build-OpenSSL-VC-64 \
ms\do_win64a \
nmake -f ms\ntdll.mak \
nmake -f ms\ntdll.mak install \
nmake -f ms\ntdll.mak test 

Build OpenSSL complete. \
Connecting libraries to a project in Visual Studio 2017 Community. \
It's very simple! \
Go to Solution Explorer - Project Properties and select: 
1. Select C/C++ next: \
C/C++ - General - Additional directories for included files - Edit - Create a line and specify the path: \
C:\Build-OpenSSL-VC-64\include \
OK 
2. Select Linker next: \
Linker - Input - Additional dependencies - Edit and specify the path: \
c:\Build-OpenSSL-VC-64\lib\ssleay32.lib \
c:\Build-OpenSSL-VC-64\lib\libeay32.lib \
OK \
Project Properties - Apply - OK \
Build and enjoy.\
\
Note: The current relase has been compiled with CUDA SDK 10.2, if you have a different release of the CUDA SDK, you may need to update CUDA SDK paths in VanitySearch.vcxproj using a text editor. 
The current nvcc option are set up to architecture starting at 3.0 capability, for older hardware, add the desired compute capabilities to the list in GPUEngine.cu properties, CUDA C/C++, Device, Code Generation.

## Linux

Intall OpenSSL.\
Intall CUDA SDK.\
Depenging on the CUDA SDK version and on your Linux distribution you may need to install an older g++ (just for the CUDA SDK).\
Edit the makefile and set up the good CUDA SDK path and appropriate compiler for nvcc. 

```
CUDA       = /usr/local/cuda-8.0
CXXCUDA    = /usr/bin/g++-4.8
```

You can enter a list of architectrure (refer to nvcc documentation) if you have several GPU with different architecture. Compute capability 2.0 (Fermi) is deprecated for recent CUDA SDK.
VanitySearch need to be compiled and linked with a recent gcc (>=7). The current release has been compiled with gcc 7.3.0.\
Go to the VanitySearch directory. 
ccap is the desired compute capability https://ru.wikipedia.org/wiki/CUDA

```
$ g++ -v
gcc version 7.3.0 (Ubuntu 7.3.0-27ubuntu1~18.04)
$ make all (for build without CUDA support)
or
$ make gpu=1 ccap=30 all
```

# License

vs-kangaroo-multi-divide is licensed under GPLv3.

