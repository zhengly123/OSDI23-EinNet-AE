`bash run.sh` produces the results in Figure 16.

`test_OpSearch` is built as a part of EinNet test.

This experiment cost about 7 hours. For a quick verification, you can change the search depth (`$(seq 1 11)` in `run.sh`) to a suitable range. We also provide example logs in `example_log`.

Note: configure the project in CMake Release mode to get a reasonalbe execution time.
