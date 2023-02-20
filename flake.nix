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
    in
    rec {

        devShells.default = pkgs.mkShell {
           venvDir = "./.venv";
           buildInputs = with pkgs; [
               pkgs.python310Packages.python
               pkgs.python310Packages.venvShellHook
               pkgs.ffmpeg
#               pkgs.libgccjit
               # this is how we add native dependencies to the shell
               # e.g. grpc libstdc++.so.6
               stdenv.cc.cc.lib
           ];
          postVenvCreation = ''
            unset SOURCE_DATE_EPOCH
            pip install -r requirements.txt --editable .
          '';

          # Now we can execute any commands within the virtual environment.
          # This is optional and can be left out to run pip manually.
          postShellHook = ''
            # allow pip to install wheels
            echo "LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib" >> .venv/bin/activate
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib 
            unset SOURCE_DATE_EPOCH
          '';
       };

      packages.nix-python =
        pkgs.python310.pkgs.buildPythonPackage rec {
         pname = "nix-python";
         version = "0.0alpha";
         format = "pyproject";

          src = ./.;

          buildInputs = with pkgs; [
               pkgs.python310Packages.setuptools
               pkgs.python310Packages.loguru
               pkgs.python310Packages.flask
               pkgs.python310Packages.torch
               pkgs.python310Packages.pandas
               # this is how we add native dependencies to the shell
               # e.g. grpc libstdc++.so.6
               stdenv.cc.cc.lib
           ];

          propagatedBuildInputs = with pkgs; [
               pkgs.python310Packages.setuptools
               pkgs.python310Packages.loguru
               pkgs.python310Packages.flask
               pkgs.python310Packages.torch
               pkgs.python310Packages.pandas
               # this is how we add native dependencies to the shell
               # e.g. grpc libstdc++.so.6
               stdenv.cc.cc.lib
           ];

          setuptoolsCheckPhase = "true";
        };

        defaultPackage = packages.nix-python;

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.nix-python}/bin/nix-python";
        };
    });
}
