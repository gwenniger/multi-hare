
Warp-ctc contains the pytorch bindings for Baidu's warp-ctc loss.

Documentation for how to deal with submodules in git:
https://git-scm.com/book/en/v2/Git-Tools-Submodulesy

When running the tests in the build foler, 
you may need to add the LD_LIBARU_PATH  to get things to work:
https://stackoverflow.com/questions/4581305/error-while-loading-shared-libraries-libboost-system-so-1-45-0-cannot-open-shai
Otherwise it may not be able to find the "libwarpctc.so" library that is build, when running test_gpu and test_cpu

This can be done with :
export LD_LIBRARY_PATH=PATH-TO-WARP-CTC-ROOT/warp-ctc/build:$LD_LIBRARY_PATH

====
Fixing the cloned repository:
1) edit the git url in  .gitmodules
2) run git submodule sync
https://stackoverflow.com/questions/913701/changing-remote-repository-for-a-git-submodule/43937092

===
Fixing problems with detached head state in the cloned repository:
https://stackoverflow.com/questions/18770545/why-is-my-git-submodule-head-detached-from-master
