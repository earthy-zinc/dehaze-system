class OptionType {
  const OptionType({this.value, this.label = '', this.children});

  final dynamic value;
  final String label;
  final List<OptionType>? children;

  factory OptionType.fromJson(Map<String, dynamic> json) => OptionType(
        value: json['value'],
        label: (json['label'] as String?) ?? '',
        children: (json['children'] as List<dynamic>?)
            ?.map((e) => OptionType.fromJson(e as Map<String, dynamic>))
            .toList(),
      );
}
