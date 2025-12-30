# """
# Plugin to fix Material search plugin compatibility with i18n.
# """
# from mkdocs.plugins import BasePlugin
# from mkdocs.config import config_options


# class MaterialSearchI18nFix(BasePlugin):
#     """Fix Material search plugin compatibility with i18n plugin."""

#     config_scheme = ()

#     def on_config(self, config, **kwargs):
#         """Fix Material search plugin translation issue."""
#         # Material search plugin tries to access translation function
#         # before i18n plugin is initialized. This plugin ensures i18n
#         # is initialized first by being placed before material/search.
#         return config
