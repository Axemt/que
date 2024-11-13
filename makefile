CONFIG_FOLDER = ~/.config/que
all:
	pip install .

clean:
	pip uninstall que -y

clean-config:
	rm $(CONFIG_FOLDER)/config.toml

clean-db:
	rm -rf $(CONFIG_FOLDER)/index.chroma

clean-all: clean clean-config
	rm -rf $(CONFIG_FOLDER)