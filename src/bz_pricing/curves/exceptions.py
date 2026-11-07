"""
Exceções customizadas para construção e gerenciamento de curvas de juros.
"""

class DataUnavailableError(ValueError):
    """
    Exceção levantada quando os dados de mercado necessários para o cálculo
    de uma curva não estão disponíveis (localmente ou para download na B3/BCB),
    e nenhum dado externo foi fornecido pelo usuário.
    """
    pass

