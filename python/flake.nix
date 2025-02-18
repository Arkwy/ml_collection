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
                (opencv4.override { enableGtk3 = true; })
              ]);
          };
        }
      );
    };
}
