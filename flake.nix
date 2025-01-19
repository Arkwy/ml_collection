{
  description = "A Nix-flake-based C/C++ development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
    }:
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
            pkgs = import nixpkgs {
              inherit system;
              overlays = [
                rust-overlay.overlays.default
                self.overlays.default
              ];
            };
          }
        );
    in
    {
      overlays.default = final: prev: {
        rustToolchain =
          let
            rust = prev.rust-bin;
          in
          if builtins.pathExists ./rust-toolchain.toml then
            rust.fromRustupToolchainFile ./rust-toolchain.toml
          else if builtins.pathExists ./rust-toolchain then
            rust.fromRustupToolchainFile ./rust-toolchain
          else
            rust.stable.latest.default.override {
              extensions = [
                "rust-src"
                "rustfmt"
              ];
            };
      };

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
                venvDir = ".venv";

                # wgpu utils (c.f. https://github.com/gfx-rs/wgpu/blob/trunk/shell.nix)
                buildInputs = with pkgs; [
                  # necessary for building wgpu in 3rd party packages (in most cases)
                  libxkbcommon
                  wayland
                  xorg.libX11
                  xorg.libXcursor
                  xorg.libXrandr
                  xorg.libXi
                  alsa-lib
                  fontconfig
                  freetype
                  shaderc
                  directx-shader-compiler
                  pkg-config
                  cmake
                  mold # could use any linker, needed for rustix (but mold is fast)

                  libGL
                  vulkan-headers
                  vulkan-loader
                  vulkan-tools
                  vulkan-tools-lunarg
                  vulkan-extension-layer
                  vulkan-validation-layers # don't need them *strictly* but immensely helpful

                  # necessary for developing (all of) wgpu itself
                  cargo-nextest
                  cargo-fuzz

                  # nice for developing wgpu itself
                  typos

                  # if you don't already have rust installed through other means,
                  # this shell.nix can do that for you with this below
                  yq # for tomlq below
                  rustup

                  # nice tools
                  gdb
                  rr
                  evcxr
                  valgrind
                  renderdoc
                ];

                packages =
                  with pkgs;
                  [
                    # c/c++
                    clang-tools
                    cmake
                    codespell
                    conan
                    cppcheck
                    doxygen
                    gtest
                    lcov
                    vcpkg
                    vcpkg-tool
                    gdb
                  ]
                  ++ [
                    # rust
                    rustToolchain
                    openssl
                    pkg-config
                    cargo-deny
                    cargo-edit
                    cargo-watch
                    rust-analyzer
                    wgsl-analyzer
                  ]
                  ++ [
                    # python
                    python312
                  ]
                  ++ (with pkgs.python312Packages; [
                    pip
                    numpy
                    matplotlib
                    venvShellHook
                  ]);

                env = {
                  # Required by rust-analyzer
                  RUST_SRC_PATH = "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
                };

                postShellHook = ''
                  export RUSTC_VERSION="$(tomlq -r .toolchain.channel rust-toolchain.toml)"
                  export PATH="$PATH:''${CARGO_HOME:-~/.cargo}/bin"
                  export PATH="$PATH:''${RUSTUP_HOME:-~/.rustup/toolchains/$RUSTC_VERSION-x86_64-unknown-linux/bin}"
                  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${builtins.toString (pkgs.lib.makeLibraryPath buildInputs)}";

                  rustup default $RUSTC_VERSION
                  rustup component add rust-src rust-analyzer
                '';

              };
        }
      );
    };
}
