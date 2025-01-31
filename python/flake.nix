{
  description = "A Nix-flake-based C/C++ development environment";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs =
    {
      self,
      nixpkgs,
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
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      allowBroken = true;
      devShells = forEachSupportedSystem (
        { pkgs }:
        {
          default = pkgs.mkShell rec {
            venvDir = ".venv";
            # buildInputs = with pkgs; [
            #   zstd
            #   zlib
            # ];
            packages =
              with pkgs;
              [
                # python
                python313
              ]
              ++ (with pkgs.python313Packages; [
                pip
                python-lsp-server
                torch
                numpy
                venvShellHook
              ]);
            # postShellHook = ''
            #   export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
            #   export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
            # '';
          };
        }
      );
    };
}
