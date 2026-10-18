"""The catalog: numpy planning columns plus lazily fetched variable fields, stored in
one single-writer KohakuVault file (see docs/concepts/data.md#the-catalog)."""

from kohakumoss_data.catalog.table import Catalog, open_catalog
from kohakumoss_data.catalog.vault import CatalogVault, is_catalog_vault

__all__ = ["Catalog", "CatalogVault", "is_catalog_vault", "open_catalog"]
