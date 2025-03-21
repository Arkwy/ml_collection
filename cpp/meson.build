project('ml_colletion', 'cpp',
  default_options: ['cpp_std=c++20'],
  meson_version: '>=1.7.0'
)

# Log level selection
log_level_map = {
  'debug': '0',
  'info': '1',
  'warning': '2',
  'error': '3'
}

log_level_str = get_option('log_level')
log_level_value = log_level_map[log_level_str]

add_project_arguments('-DLOG_LEVEL=' + log_level_value, language: 'cpp')


# Common cpp files
common_sources = []


# Algorithms specific cpp files
algo_sources = {'pso' : [
  'pso/src/main.cpp',
  'pso/src/ops/fill.cpp',
  'pso/src/utils/hip_timer.cpp',
  'pso/src/utils/logger.cpp'
]}


# Create an executable for each algorithm
binaries = []
foreach name, sources : algo_sources
  bin_name = name
  bin = executable(bin_name, [sources,  common_sources],
    cpp_args: ['-O2'],
    link_args: ['-lrocrand'],
  )
  binaries += bin
endforeach
