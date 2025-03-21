{
  description = "A Nix-flake-based C/C++ development environment";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs }:
        {
          default =
            pkgs.mkShell.override
              {
                # Override stdenv in order to change compiler:
                stdenv = pkgs.clangStdenv;
              }
              rec {
                packages = with pkgs; [
                  # c/c++
                  clang-tools
                  codespell
                  conan
                  cppcheck
                  doxygen
                  gtest
                  lcov
                  vcpkg
                  vcpkg-tool
                  gdb
                  valgrind
                  meson
                  ninja

                  # correct lsp support
                  bear

                  # rocm
                  rocmPackages.clr
                  rocmPackages.rocm-smi
                  rocmPackages.rocrand
                  rocmPackages.rocgdb
                ];

                # Environment variables required for ROCm (PYTHONWARNINGS ignored because some appears with seemingly no impact during builds with hipcc)
                shellHook = ''
                  export ROCM_PATH=${pkgs.rocmPackages.clr};
                  export PYTHONWARNINGS="ignore";
                '';
              };
        }
      );
    };
}
