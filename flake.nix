{
  description = "Nix flake environment for a sample python project";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/release-22.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self , flake-utils , nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
    let
        pkgs = import nixpkgs { inherit system; };
        pythonEnv = pkgs.python3.withPackages(ps: with ps; [
            pip virtualenv wheel
        ]);
    in
    {
        devShells.default = pkgs.mkShell {
           buildInputs = with pkgs; [
               pythonEnv
               # this is how we add native dependencies to the shell
               # e.g. grpc libstdc++.so.6
               stdenv.cc.cc.lib
           ];

           shellHook = ''
               # make sure that python could load that lib
               export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/:$LD_LIBRARY_PATH
           '';
       };
    });
}
